from torchvision import models, transforms
import torch
from torch.autograd import Variable

model = models.resnet18(pretrained=True)

layer = model._modules.get('avgpool')

model.eval()

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()


def get_vector(img):
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    my_embedding = torch.zeros(1, 512)

    def copy_data(m, i, o):
        my_embedding.copy_(o.data.view(o.data.size(0), -1))

    h = layer.register_forward_hook(copy_data)
    model(t_img)
    h.remove()
    return my_embedding.numpy()


def main():
    print(get_vector('image.jpg'))


if __name__ == '__main__':
    main()
