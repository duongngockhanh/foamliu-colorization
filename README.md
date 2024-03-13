# vgg-clas-colorization
Implementation of Colorful Image Colorization by Richard Zhang.

## Reference
[Code](https://github.com/foamliu/Character-Coloring)

## Run session
```
!pip install wandb
from kaggle_secrets import UserSecretsClient
import wandb
user_secrets = UserSecretsClient()
wandb_api = user_secrets.get_secret("wandb_api") 
wandb.login(key=wandb_api)
```
```
!git clone https://github.com/duongngockhanh/vgg-clas-colorization.git
```
```
!python /kaggle/working/vgg-clas-colorization/train.py
```
