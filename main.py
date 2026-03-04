import os
from TRM_net_v2 import denoising_model
from new_funciton import data_load, normalize, loss_function1
import torch
import torch.utils.data as Data
import torch.nn as nn
import time
from datetime import datetime

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

batch_size = 64
epochs = 200 # shoulian VAE:500, TRM:1000 STILL DOWN
learning_rate = 0.001
weight_decay = 0

def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
train_data = torch.Tensor(data_load('train_data'))
train_label = torch.Tensor(data_load('train_label'))

test_data = torch.Tensor(data_load('test_data'))
test_label = torch.Tensor(data_load('test_label'))


dataset_train = Data.TensorDataset(train_data, train_label)
dataset_test = Data.TensorDataset(test_data, test_label)

train_loader = Data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = Data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"training_logs/{timestamp}"
os.makedirs(log_dir, exist_ok=True)

history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }

csv_path = os.path.join(log_dir, "training_log.csv")

best_val_acc = 0.0

def main():

    net = denoising_model()

    net.train()
    torch.autograd.set_detect_anomaly(False)
    device = torch.device("cuda:0")
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    net = net.to(device)
    min_loss = 1


    for epoch in range(epochs):
        since = time.time()
        total_loss = 0

        for step, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()

            data = torch.Tensor(data).view(batch_size,1024)
            label = torch.Tensor(label).view(batch_size,1024)

            #
            data = torch.log10(torch.Tensor(abs(data)))
            label = torch.log10(torch.Tensor(abs(label)))
            data, label, scale = normalize(data, label)


            net.train()

            data = torch.Tensor(data).view(batch_size,1,1024).to(device)
            label = torch.Tensor(label).view(batch_size,1024).to(device)

            pre_label = net(data)

            pre_label = pre_label.view(64,1024)
            label = label.to(device).view(64,1024)

            loss = loss_function1(pre_label, label)

            loss.backward()

            grad_norm = get_grad_norm(net)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)

            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        val_loss = 0.0

        with torch.no_grad():
            for step_, (test_data_, test_label_) in enumerate(test_loader):
                net.eval()
                test_data_ = torch.log10(torch.Tensor(test_data_))
                test_label_ = torch.log10(torch.Tensor(test_label_))

                test_data_, test_label_, _ = normalize(test_data_, test_label_)
                test_data_ = test_data_.view(64, 1,1024).to(device)
                test_label_ = test_label_.view(64, 1024).to(device)

                pre_label_ = net(test_data_)
                Val_loss = loss_function1(pre_label_, test_label_)
                val_loss += Val_loss.item()



        time_elapsed = time.time() - since
        Average_loss = total_loss / len(train_loader)

        print(f"{epoch + 1} epoch's loss：{Average_loss:.8f} | Time-consuming：{time_elapsed:.1f} | lr:{optimizer.param_groups[0]['lr']}")
        if Average_loss < min_loss:
            min_loss = Average_loss
            torch.save(net, "HTRMnet.pth")

        ## train data save
        # with open(csv_path, 'a', newline='') as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #     writer.writerow({
        #         'epoch': epoch + 1,
        #         'train_loss': f"{Average_loss:.6f}",
        #         'val_loss': f"{val_loss/len(test_loader):.6f}",
        #         'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
        #         'timestamp': f"{time_elapsed:.1f}"
        #     })


if __name__ == "__main__":
    main()


