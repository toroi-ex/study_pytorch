import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import math
import random
import os


from custum import CustomDataset
from custum import data_transforms
from dataset import Mydatasets
from Discriminator import Discriminator
from Generator import Generator


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), x.shape[1], x.shape[2], x.shape[3])
    return x


def preserve_result_img(img, dir, filename, epoch):
    os.makedirs(os.path.join(dir, '{epoch}'.format(epoch=int(epoch))), exist_ok=True)
    value = int(math.sqrt(batch_size))
    pic = to_img(img.cpu().data)
    pic = torchvision.utils.make_grid(pic, nrow=value) #バッチサイズに数だけグリッドを分割する
    save_image(pic, os.path.join(dir,"{epoch}/{filename}.png".format(epoch=int(epoch),filename=filename)))


def model_init(net, input, output, model_path, device):
    model = net(input, output).to(device)
    if pretrained:
        param = torch.load(model_path)
        model.load_state_dict(param)
    return model


def reset_model_grad(G1, G2, D1, D2):
    G1.zero_grad()
    G2.zero_grad()
    D1.zero_grad()
    D2.zero_grad()


def main(fold_name1, fold_name2):
    dataset = CustomDataset(root=train_data_dir, fold1=fold_name1, fold2=fold_name2, transform=data_transforms)

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    print("dataload conplete")

    # もしGPUがあるならGPUを使用してないならCPUを使用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("GPU:",device)

    if pretrained:
        G1 = model_init(Generator, 3, 3, pretrained_model_file_name_list[0] + '.pth', device)
        G2 = model_init(Generator, 3, 3, pretrained_model_file_name_list[1] + '.pth', device)
        D1 = model_init(Discriminator, 3, 64, pretrained_model_file_name_list[2] + '.pth', device)
        D2 = model_init(Discriminator, 3, 64, pretrained_model_file_name_list[3] + '.pth', device)
    else:
        G1 = model_init(Generator, 3, 3, project_root + pretrained_model_file_name_list[0] + '.pth', device)
        G2 = model_init(Generator, 3, 3, project_root + pretrained_model_file_name_list[1] + '.pth', device)
        D1 = model_init(Discriminator, 3, 64, project_root + pretrained_model_file_name_list[2] + '.pth', device)
        D2 = model_init(Discriminator, 3, 64, project_root + pretrained_model_file_name_list[3] + '.pth', device)

    L1Loss = nn.L1Loss()
    MSELoss = nn.MSELoss()

    optimizerG1 = torch.optim.Adam(G1.parameters(), lr=learning_rate, betas=(beta1, 0.999), weight_decay=1e-5)
    optimizerD2 = torch.optim.Adam(D2.parameters(), lr=learning_rate, betas=(beta1, 0.999), weight_decay=1e-5)
    optimizerD1 = torch.optim.Adam(D1.parameters(), lr=learning_rate, betas=(beta1, 0.999), weight_decay=1e-5)
    optimizerG2 = torch.optim.Adam(G2.parameters(), lr=learning_rate, betas=(beta1, 0.999), weight_decay=1e-5)

    for epoch in range(num_epochs):
        itr = 0
        loss_list_g1 = []
        loss_list_g2 = []
        loss_list_d1 = []
        loss_list_d2 = []
        for data, data2, file_path1, file_path2 in dataloader:
            itr = itr + 1
            image1 = data.to(device)  # 本物画像
            image2 = data2.to(device)  # 本物画像
            sample_size = image1.size(0)  # 画像枚数
            real_target = torch.full((sample_size, 1, 1), random.uniform(1, 1), device=device)  # 本物ラベル
            fake_target = torch.full((sample_size, 1, 1), random.uniform(0, 0), device=device)  # 偽物ラベル

            # ------Discriminatorの学習-------
            reset_model_grad(G2, G1, D2, D1)

            fake_image2 = G2(image1)  # 生成画像
            output = D2(fake_image2)  # 生成画像に対するDiscriminatorの結果
            adversarial_nogi_loss_fake = MSELoss(output, real_target)  # Discriminatorの出力結果と正解ラベルとのMSELoss

            image1_image2_image1 = G1(fake_image2)
            image2_image1_image2 = G2(G1(image2))  # 生成画像)
            loss_image1_image2_image1 = L1Loss(image1_image2_image1, image1)
            loss_image2_image1_image2 = L1Loss(image2_image1_image2, image2)

            #identity loss
            identify_normal = L1Loss(G2(image2), image2)

            loss_g_2 = identify_normal * cycle_late + loss_image2_image1_image2 * cycle_late + adversarial_nogi_loss_fake + loss_image1_image2_image1 * cycle_late  # 二つの損失をバランスを考えて加算
            loss_g_2.backward()  # 誤差逆伝播
            # loss_list_g2.append(loss_g_2)
            optimizerG2.step()  # Generatorのパラメータ更新

            reset_model_grad(G2, G1, D2, D1)  # 勾配情報の初期化

            fake_image1 = G1(image2)  # 生成画像
            output = D1(fake_image1)  # 生成画像に対するDiscriminatorの結果
            adversarial_normal_loss_fake = MSELoss(output, real_target)  # Discriminatorの出力結果と正解ラベルとのBCELoss

            image2_image1_image2 = G2(fake_image1)
            image1_image2_image1 = G1(G2(image1))

            loss_image2_image1_image2 = L1Loss(image2_image1_image2, image2)
            loss_image1_image2_image1 = L1Loss(image1_image2_image1, image1)

            identify_nogi = L1Loss(G1(image1), image1)

            loss_g_1 = identify_nogi * cycle_late + loss_image1_image2_image1 * cycle_late + adversarial_normal_loss_fake + loss_image2_image1_image2 * cycle_late  # 二つの損失をバランスを考えて加算
            loss_g_1.backward()  # 誤差逆伝播
            # loss_list_g1.append(loss_g_1)
            optimizerG1.step()  # Generatorのパラメータ更新

            reset_model_grad(G2, G1, D2, D1)  # 勾配情報の初期化

            fake_image2 = G2(image1)  # 生成画像
            output = D2(fake_image2)  # 生成画像に対するDiscriminatorの結果
            adversarial_nogi_loss_fake = MSELoss(output, fake_target)  # Discriminatorの出力結果と正解ラベルとのBCELoss

            output = D2(image2)  # 生成画像に対するDiscriminatorの結果
            adversarial_nogi_loss_real = MSELoss(output, real_target)  # Discriminatorの出力結果と正解ラベルとのBCELoss

            loss_d_2 = (adversarial_nogi_loss_fake + adversarial_nogi_loss_real) * 1  # 単純に加算
            loss_d_2.backward()  # 誤差逆伝播
            # loss_list_d2.append(loss_d_2)
            optimizerD2.step()  # Discriminatorのパラメータ更新

            # 勾配情報の初期化
            reset_model_grad(G2, G1, D2, D1)

            fake_image1 = G1(image2)  # 生成画像
            output = D1(fake_image1)  # 生成画像に対するDiscriminatorの結果
            adversarial_normal_loss_fake = MSELoss(output, fake_target)  # Discriminatorの出力結果と正解ラベルとのBCELoss

            output = D1(image1)  # 生成画像に対するDiscriminatorの結果
            adversarial_normal_loss_real = MSELoss(output, real_target)  # Discriminatorの出力結果と正解ラベルとのBCELoss

            loss_d_1 = (adversarial_normal_loss_fake + adversarial_normal_loss_real) * 1  # 単純に加算
            loss_d_1.backward()  # 誤差逆伝播
            # loss_list_d1.append(loss_d_1)
            optimizerD1.step()  # Discriminatorのパラメータ更新

            fake_image2 = G2(image1)  # 生成画像
            fake_image1 = G1(image2)  # 生成画像
            img_list = [image1, image2, fake_image1, fake_image2, image1_image2_image1, image2_image1_image2]
            img_file_name_list = ['image1', 'image2', 'fake_image1', 'fake_image2', 'image1_image2_image1',
                                  'image2_image1_image2']

            print(
                "train epoch: [{0:d}/{1:d}][{2:d}/{3:d}] Loss_g1: {4:.4f} Loss_g2: {5:.4f} Loss_d1: {6:.4f} Loss_d2: {7:.4f}".format(
                    epoch+1, num_epochs, itr, len(dataloader), loss_g_1, loss_g_2, loss_d_1, loss_d_2))

            if itr == len(dataloader.dataset) and save_img == True:
                print(epoch, itr, len(dataloader), loss_g_1, loss_g_2, loss_d_1, loss_d_2)
                for i in range(len(img_list)):
                    preserve_result_img(img_list[i], project_root, img_file_name_list[i], epoch+1)

        # path = project_root + "/loss_G1_{}.txt".format(epoch)
        # with open(path, mode='w') as f:
        #     for loss in loss_list_g1:
        #         f.write("{}\n".format(loss))
        #
        # path = project_root + "/loss_D1_{}.txt".format(epoch)
        # with open(path, mode='w') as f:
        #     for loss in loss_list_d1:
        #         f.write("{}\n".format(loss))
        # path = project_root + "/loss_G2_{}.txt".format(epoch)
        # with open(path, mode='w') as f:
        #     for loss in loss_list_g2:
        #         f.write("{}\n".format(loss))
        #
        # path = project_root + "/loss_D2_{}.txt".format(epoch)
        # with open(path, mode='w') as f:
        #     for loss in loss_list_d2:
        #         f.write("{}\n".format(loss))

        # モデルを保存
        model_list = [G1, G2, D1, D2]
        for i in range(len(model_list)):
            torch.save(model_list[i].state_dict(), project_root + '/' + output_model_file_name_list[i] + '.pth')


if __name__ == '__main__':
    beta1 = 0.5
    cycle_late = 1  # L1LossとadversarilLossの重要度を決定する係数
    num_epochs = 20  # エポック数 5の場合は8 1の場合は10
    batch_size = 1  # バッチサイズ
    learning_rate = 1e-4  # 学習率
    pretrained = True  # 事前に学習したモデルがあるならそれを使う
    # pretrained_model_file_name_list = ['G1_B4', 'G2_B4', 'D1_B4', 'D2_B4']
    pretrained_model_file_name_list = ['/home/toui/PycharmProjects/toui_pytorch/cycle_gan2/result/G1_B4',
                                       '/home/toui/PycharmProjects/toui_pytorch/cycle_gan2/result/G2_B4',
                                       '/home/toui/PycharmProjects/toui_pytorch/cycle_gan2/result/D1_B4',
                                       '/home/toui/PycharmProjects/toui_pytorch/cycle_gan2/result/D2_B4']
    output_model_file_name_list = ['G1_B4', 'G2_B4', 'D1_B4', 'D2_B4']
    save_img = True  # ネットワークによる生成画像を保存するかどうかのフラグ

    #powoer device
    # project_root = '/home/toui/PycharmProjects/toui_pytorch/cycle_gan2/result'
    # train_data_dir = '/home/toui/デスクトップ/ori/add_data_gan/broken_mix'
    # main("broken_03", "broken_06")

    #sankyu
    project_root = '/home/toui/PycharmProjects/toui_pytorch/cycle_gan2/sensei_help/training_result'
    train_data_dir = '/home/toui/デスクトップ/sankyu/input'
    main("img", "img1")




    #cell
    # dir_class_list = ["bright", "cytoplasm", "merged", "nucleus"]
    # dir_class = "/"+dir_class_list[3]
    # project_root="/home/toui/PycharmProjects/toui_pytorch/cycle_gan2/cell/training_result" + dir_class
    # train_data_dir="/home/toui/デスクトップ/cell_gan/input" + dir_class
    # main("HL", "MCF")

    print("train complete")
