import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as T
torch.backends.cudnn.benchmark= True  # Provides a speedup

from DAGeoLoc import DAGeoLocNet
from DA_TrainDataset import DATrainDataset
import test
import util
import parser
import commons
import cosface_loss
import augmentations
import network
from test_dataset import TestDataset
from train_dataset import TrainDataset

args = parser.parse_arguments()
start_time = datetime.now()
output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)
commons.setup_logging(output_folder, console="debug")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {output_folder}")

#### Model
model = DAGeoLocNet(args.backbone, args.fc_output_dim, alpha = 0.05)

logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

if args.resume_model != None:
    logging.debug(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    model.load_state_dict(model_state_dict)

model = model.to(args.device).train()

#### Optimizer
criterion = torch.nn.CrossEntropyLoss()
model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#### Datasets UTM labels
groups = [DATrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                       current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]
# Each group has its own classifier, which depends on the number of classes in the group
classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]
classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers]

#How many classes and images for the class label prediction
logging.info(f"Using {len(groups)} groups")
logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold)
test_ds = TestDataset(args.test_set_folder, queries_folder="queries_v1",
                      positive_dist_threshold=args.positive_dist_threshold)
logging.info(f"Validation set: {val_ds}")
logging.info(f"Test set: {test_ds}")

# Dataset day label (1,1,1)
groups_day = [DATrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                       current_group=n, min_images_per_class=args.min_images_per_class, day = True) for n in range(args.groups_num)]
# Each group has its own classifier, which depends on the number of classes in the group
classifiers_day = [cosface_loss.MarginCosineProduct(2, len(group)) for group in groups_day]
classifiers_optimizers_day = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers_day]

#How many classes and images for the day domain label prediction
logging.info(f"Using {len(groups_day)} groups")
logging.info(f"The {len(groups_day)} groups have respectively the following number of classes {[len(g) for g in groups_day]}")
logging.info(f"The {len(groups_day)} groups have respectively the following number of images {[g.get_images_num() for g in groups_day]}")

logging.info(f"PROVAPROVA DAY {groups_day[0]} ")

#Dataset night label (0,0,0)
groups_night = [DATrainDataset(args, "/content/drive/MyDrive/MLDL2022/Project3/CosPlace/datasets/toky_night/night", M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                       current_group=n, min_images_per_class=args.min_images_per_class, night = True) for n in range(args.groups_num)]
# Each group has its own classifier, which depends on the number of classes in the group
classifiers_night = [cosface_loss.MarginCosineProduct(2, len(group)) for group in groups_night]
classifiers_optimizers_night = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers_night]

#How many classes and images for the night domain label prediction
logging.info(f"Using {len(groups_night)} groups")
logging.info(f"The {len(groups_night)} groups have respectively the following number of classes {[len(g) for g in groups_night]}")
logging.info(f"The {len(groups_night)} groups have respectively the following number of images {[g.get_images_num() for g in groups_night]}")

#### Resume
if args.resume_train:
    model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num = \
        util.resume_train(args, output_folder, model, model_optimizer, classifiers, classifiers_optimizers)
    model = model.to(args.device)
    epoch_num = start_epoch_num - 1
    logging.info(f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.1f} from checkpoint {args.resume_train}")
else:
    best_val_recall1 = start_epoch_num = 0

#### Train / evaluation loop
logging.info("Start training ...")
logging.info(f"There are {len(groups[0])} classes for the first group, " +
             f"each epoch has {args.iterations_per_epoch} iterations " +
             f"with batch_size {args.batch_size}, therefore the model sees each class (on average) " +
             f"{args.iterations_per_epoch * args.batch_size / len(groups[0]):.1f} times per epoch")


if args.augmentation_device == "cuda":
    gpu_augmentation = T.Compose([
            augmentations.DeviceAgnosticColorJitter(brightness=args.brightness,
                                                    contrast=args.contrast,
                                                    saturation=args.saturation,
                                                    hue=args.hue),
            augmentations.DeviceAgnosticRandomResizedCrop([512, 512],
                                                          scale=[1-args.random_resized_crop, 1]),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

if args.use_amp16:
    scaler = torch.cuda.amp.GradScaler()

#from torch.utils.data import DataLoader
for epoch_num in range(start_epoch_num, args.epochs_num):
    
    #### Train
    epoch_start_time = datetime.now()
    # Select classifier and dataloader according to epoch
    current_group_num = epoch_num % args.groups_num
    
    #class label clasifier
    classifiers[current_group_num] = classifiers[current_group_num].to(args.device)
    util.move_to_device(classifiers_optimizers[current_group_num], args.device)
    
    #day domain classifier
    classifiers_day[current_group_num] = classifiers_day[current_group_num].to(args.device)
    util.move_to_device(classifiers_optimizers_day[current_group_num], args.device)
    
    #night domain classifier
    classifiers_night[current_group_num] = classifiers_night[current_group_num].to(args.device)
    util.move_to_device(classifiers_optimizers_night[current_group_num], args.device)
    
    # classic dataloader    
    dataloader = commons.InfiniteDataLoader(groups[current_group_num], num_workers=args.num_workers,
                                            batch_size=args.batch_size, shuffle=True,
                                            pin_memory=(args.device=="cuda"), drop_last=True)
    
    dataloader_iterator = iter(dataloader)
    model = model.train()
    
    # day dataloader
    dataloader_day = commons.InfiniteDataLoader(groups_day[current_group_num], num_workers=args.num_workers,
                                            batch_size=1, shuffle=True,
                                            pin_memory=(args.device=="cuda"), drop_last=True)
    dataloader_iterator_day = iter(dataloader_day)
    
    # night dataloader
    dataloader_night = commons.InfiniteDataLoader(groups_night[current_group_num], num_workers=args.num_workers,
                                            batch_size=1, shuffle=True,
                                            pin_memory=(args.device=="cuda"), drop_last=True)
    
    dataloader_iterator_night = iter(dataloader_night)
    
    
    logging.info(f"Dataloader CLASSIC: {len(dataloader)}")
    logging.info(f"Dataloader DAY: {len(dataloader_day)}")
    logging.info(f"Dataloader NIGHT: {len(dataloader_night)}")
    

    epoch_losses = np.zeros((0,1), dtype=np.float32)
    for iteration in tqdm(range(args.iterations_per_epoch), ncols=100):        
        # images, target UTM labels
        images, targets, _ = next(dataloader_iterator)
        images, targets = images.to(args.device), targets.to(args.device)
        
        #images, target day domain labels
        images_day, targets_day, _ = next(dataloader_iterator_day)
        images_day, targets_day = images_day.to(args.device), targets_day.to(args.device)
        
        #images,target night domain labels
        images_night, targets_night, _ = next(dataloader_iterator_night)
        images_night, targets_night = images_night.to(args.device), targets_night.to(args.device)
        
        if args.augmentation_device == "cuda":
            images = gpu_augmentation(images)
            images_day = gpu_augmentation(images_day)
            images_night = gpu_augmentation(images_night)
            
        
        model_optimizer.zero_grad()
        classifiers_optimizers[current_group_num].zero_grad()
        
        if not args.use_amp16:
            
            # loss classic
            descriptors = model(images)
            output = classifiers[current_group_num](descriptors, targets)
            loss = criterion(output, targets)
            loss.backward()
            
            #loss day
            descriptors_day = model(images_day, alpha = 0.05)
            output_day = classifiers_day[current_group_num](descriptors_day, targets_day)
            loss_day = criterion(output_day, targets_day)
            
            #loss night
            descriptors_night = model(images_night, alpha = 0.05)
            output_night = classifiers_night[current_group_num](descriptors_night, targets_night)
            loss_night = criterion(output_night, targets_night)
            
            #loss domain 
            loss_domain = loss_night + loss_day
            loss_domain.backward()

            epoch_losses = np.append(epoch_losses, loss.item())
            epoch_losses = np.append(epoch_losses, loss_domain.item())
            
            del loss,loss_night,loss_day, loss_domain, output, output_day, output_night, images, images_day, images_night
            model_optimizer.step()
            classifiers_optimizers[current_group_num].step()
        
        else:  # Use AMP 16
            with torch.cuda.amp.autocast():
                #loss classic
                descriptors = model(images)
                output = classifiers[current_group_num](descriptors, targets)
                loss = criterion(output, targets)

                #loss day
                descriptors_day = model(images_day, alpha = 0.05)
                output_day = classifiers_day[current_group_num](descriptors_day, targets_day)
                loss_day = criterion(output_day, targets_day)
            
                #loss night
                descriptors_night = model(images_night, alpha = 0.05)
                output_night = classifiers_night[current_group_num](descriptors_night, targets_night)
                loss_night = criterion(output_night, targets_night)

                #loss domain 
                loss_domain = loss_night + loss_day
                

            scaler.scale(loss).backward()
            scaler.scale(loss_domain).backward()

            epoch_losses = np.append(epoch_losses, loss.item())
            epoch_losses = np.append(epoch_losses, loss_domain.item())

            del loss,loss_night,loss_day, loss_domain, output, output_day, output_night, images, images_day, images_night
            scaler.step(model_optimizer)
            scaler.step(classifiers_optimizers[current_group_num])
            scaler.update()
    
    classifiers[current_group_num] = classifiers[current_group_num].cpu()
    util.move_to_device(classifiers_optimizers[current_group_num], "cpu")
    
    logging.debug(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                 f"loss = {epoch_losses.mean():.4f}")
    
    #### Evaluation
    recalls, recalls_str = test.test(args, val_ds, model)
    logging.info(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, {val_ds}: {recalls_str[:20]}")    
    ## DA CAMBIARE E RIMETTERE COME PRIMA
    is_best = recalls[0] > best_val_recall1
    best_val_recall1 = max(recalls[0], best_val_recall1)
    # Save checkpoint, which contains all training parameters
    util.save_checkpoint({"epoch_num": epoch_num + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model_optimizer.state_dict(),
        "classifiers_state_dict": [c.state_dict() for c in classifiers],
        "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
        "best_val_recall1": best_val_recall1
    }, is_best, output_folder)


logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

best_model_state_dict = torch.load(f"{output_folder}/best_model.pth")
model.load_state_dict(best_model_state_dict)

#### Test best model on test set v1 san francisco
logging.info(f"Now testing on the test set: {test_ds}")
recalls, recalls_str = test.test(args, test_ds, model)
logging.info(f"{test_ds}: {recalls_str}")

 # Test on tokyo queries
test_ds_tokyo = TestDataset("/content/drive/MyDrive/MLDL2022/Project3/CosPlace/datasets/tokyo_xs/test", queries_folder="queries_v1",
                       positive_dist_threshold=args.positive_dist_threshold)
logging.info(f"Now testing on the test set TOKYO: {test_ds_tokyo}")
recalls2, recalls_str2 = test.test(args, test_ds_tokyo, model)
logging.info(f"{test_ds_tokyo}: {recalls_str2}")

#Test on tokyo night queries
test_tokyo_night = TestDataset("/content/drive/MyDrive/MLDL2022/Project3/CosPlace/datasets/toky_night", queries_folder="night",
                      positive_dist_threshold=args.positive_dist_threshold)
logging.info(f"Now testing on the test set TOKYO NIGHT: {test_tokyo_night}")
recalls3, recalls_str3 = test.test(args, test_tokyo_night, model)
logging.info(f"{test_tokyo_night}: {recalls_str3}")

logging.info("Experiment finished (without any errors)")
