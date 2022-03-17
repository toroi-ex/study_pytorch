import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import math
import os

# from .custum import CustomDataset
from custum import CustomDataset, CustomDatasetTestA, CustomDatasetTestB
from custum import data_val_transforms

from Generator import Generator


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), x.shape[1], x.shape[2], x.shape[3])
    return x


def preserve_result_img(img, dir, filename, fake):
    os.makedirs(os.path.join(dir, '{img}'.format(img=fake)), exist_ok=True)
    value = int(math.sqrt(batch_size))
    pic = to_img(img.cpu().data)
    pic = torchvision.utils.make_grid(pic, nrow=value)
    save_image(pic, os.path.join(dir, "{fake_dir}/{filename}".format(fake_dir=fake, fake=fake, filename=filename)))


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


def test(fold_name1, fold_name2):
    # datasetA = CustomDatasetTestA(root=train_data_dir, fold1=fold_name1, transform=data_val_transforms)
    #
    # datasetB = CustomDatasetTestB(root=train_data_dir, fold2=fold_name2, transform=data_val_transforms)
    #
    # dataloaderA = torch.utils.data.DataLoader(dataset=datasetA, batch_size=batch_size, shuffle=True)
    #
    # dataloaderB = torch.utils.data.DataLoader(dataset=datasetB, batch_size=batch_size, shuffle=True)

    dataset = CustomDataset(root=train_data_dir, fold1=fold_name1, fold2=fold_name2, transform=data_val_transforms)

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    print("dataload conplete")

    # もしGPUがあるならGPUを使用してないならCPUを使用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("GPU:", device)

    G1 = model_init(Generator, 3, 3, project_root + '/' + pretrained_model_file_name_list[0] + '.pth', device)
    G2 = model_init(Generator, 3, 3, project_root + '/' + pretrained_model_file_name_list[1] + '.pth', device)

    # データを片方づつ読み込む場合に使う
    # with torch.no_grad():
    #     for dataA in dataloaderA: # dataAに一緒にimage1とfilepathを格納しないとエラーが起こる
    #         image1, file_path1 = dataA
    #
    #         image1 = image1.to(device)
    #
    #         # dataloaderのやつはタプルで返されるため、いらない文字を取り除く必要がある
    #         image_name = [file_path1[0].replace("(","").replace("'","").replace(")","").replace(",","")]
    #
    #         fake_image2 = G2(image1)
    #
    #         save_image_list = [fake_image2]
    #         save_image_name_list = ["fake2"]   #fake1がimage1、fake2がimage2
    #
    #         for i in range(len(save_image_list)):
    #             preserve_result_img(save_image_list[i], result_dir, image_name[i], save_image_name_list[i])
    #
    #     for dataB in dataloaderB:
    #         image2, file_path2 = dataB
    #
    #         image2 = image2.to(device)
    #
    #         # dataloaderのやつはタプルで返されるため、いらない文字を取り除く必要がある
    #         image_name = [file_path2[0].replace("(","").replace("'","").replace(")","").replace(",","")]
    #
    #         fake_image1 = G1(image2)
    #
    #         save_image_list = [fake_image1]
    #         save_image_name_list = ["fake1"]   #fake1がimage1、fake2がimage2
    #
    #         for i in range(len(save_image_list)):
    #             preserve_result_img(save_image_list[i], result_dir, image_name[i], save_image_name_list[i])

    # データを両方一緒に読み込む場合に使う
    with torch.no_grad():
        for data, data1, file_path1, file_path2 in dataloader:

            image1 = data.to(device)
            image2 = data1.to(device)

            # dataloaderのやつはタプルで返されるため、いらない文字を取り除く必要がある
            # image_name = [file_path1[0].replace("(","").replace("'","").replace(")","").replace(",",""),
            #               file_path2[0].replace("(","").replace("'","").replace(")","").replace(",","")]

            file_name1 = file_path1[0].replace("(","").replace("'","").replace(")","").replace(",","")
            file_name2 = file_path2[0].replace("(","").replace("'","").replace(")","").replace(",","")

            image_name = ["ori_"+file_name2, "ori_"+file_name1]
            print(image_name)

            fake_image1 = G1(image2)
            fake_image2 = G2(image1)

            save_image_list = [fake_image1, fake_image2]
            save_image_name_list = ["fake1", "fake2"]   #fake1がimage1、fake2がimage2

            for i in range(len(save_image_list)):
                preserve_result_img(save_image_list[i], result_dir, image_name[i], save_image_name_list[i])


if __name__ == '__main__':

    batch_size = 1  # バッチサイズ
    learning_rate = 1e-4  # 学習率
    pretrained = True  # 事前に学習したモデルがあるならそれを使う
    pretrained_model_file_name_list = ['G1_B4', 'G2_B4', 'D1_B4', 'D2_B4']

    save_img = True  # ネットワークによる生成画像を保存するかどうかのフラグ

    # power device
    # project_root = '/home/toui/PycharmProjects/toui_pytorch/cycle_gan2/result/'
    # train_data_dir = '/home/toui/デスクトップ/ori/gan_test_img'
    # result_dir = '/home/toui/デスクトップ/ori/cyclegan_result'
    # test("broken_03", "broken_06")

    # # sankyu
    project_root = '/home/toui/PycharmProjects/toui_pytorch/cycle_gan2/sensei_help/training_result'
    train_data_dir = '/home/toui/デスクトップ/sankyu/input'
    result_dir = '/home/toui/デスクトップ/sankyu/output'
    test("img", "img1")

    # cell
    # dir_class_list = ["bright", "cytoplasm", "merged", "nucleus"]
    # dir_class = "/"+dir_class_list[0]
    # project_root = "/home/toui/PycharmProjects/toui_pytorch/cycle_gan2/cell/training_result" + dir_class
    # train_data_dir = "/home/toui/デスクトップ/cell_gan/input" + dir_class
    # result_dir = "/home/toui/デスクトップ/cell_gan/output" + dir_class
    # test("HL", "MCF")


    print("test complete")
