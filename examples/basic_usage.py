import med_dataloader as med_dl

# cd to examples folder

dl = med_dl.DataLoader(imgA_label="CBCT",
                       imgB_label="CT",
                       data_dir="Test_Dataset")

train_ds, valid_ds, test_ds = med_dl.generate_dataset(data_loader=dl,
                                                      percentages=[
                                                          0.8, 0.1, 0.1],
                                                      batch_size=3,
                                                      input_size=256)
