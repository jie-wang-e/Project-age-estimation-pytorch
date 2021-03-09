import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils


class MODEL(nn.Module):
    def __init__(self, model_name="se_resnext50_32x4d", num_classes_age=101, 
                  num_classes_gender=2, pretrained="imagenet"):
        super(MODEL, self).__init__()
        img_model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
        img_model.avg_pool = nn.AdaptiveAvgPool2d(1)
        encoder_size = img_model.last_linear.in_features
        list_layer = list(img_model.children())[:-1]
        self.encoder = nn.Sequential(*list_layer)

        self.head1 = nn.Sequential(
          nn.Linear(encoder_size, 500),
          nn.ReLU(),
          nn.Linear(500, num_classes_age)
        )

        self.head2 = nn.Sequential(
          nn.Linear(encoder_size, 500),
          nn.ReLU(),
          nn.Linear(500, num_classes_gender)
        )

    def forward(self, x):
        shared_features = self.encoder(x)
        shared_features = shared_features.squeeze()
        out1 = self.head1(shared_features)
        # print("yes out_1")
        out2 = self.head2(shared_features)
        # print("yes out_2")
        return out1, out2


def main():
    model = MODEL()
    print(model)


if __name__ == '__main__':
    main()