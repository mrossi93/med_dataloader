import med_dataloader as med_dl

# cd to examples folder

dl = med_dl.DataLoader(imgA_label="CBCT",
                       imgB_label="CT",
                       data_dir="Test_Dataset_mha")

train_ds, valid_ds, test_ds = med_dl.generate_dataset(data_loader=dl,
                                                      input_size=256,
                                                      percentages=[
                                                          0.8, 0.1, 0.1],
                                                      batch_size=3,
                                                      norm_bounds=[-1024.0,
                                                                   3200.0],
                                                      random_crop_size=128,
                                                      random_rotate=True,
                                                      random_flip=True
                                                      )
