import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F

from config import device, weights_1, weights_2, q_ab, epsilon, T


def categorical_crossentropy_color_1(y_pred, y_true):
    # multiply y_true by weights
    y_true = y_true * weights_1

    cross_ent = torch.nn.functional.cross_entropy(y_pred, y_true.argmax(dim=1))
    cross_ent = torch.mean(cross_ent)

    return cross_ent

def categorical_crossentropy_color_2(y, gt):  
    # calculate loss
    loss_tmp = - gt * torch.log(y + 1e-10)
    loss_tmp_perpix = torch.sum(loss_tmp, axis = 1)
    max_idx_perpix = torch.argmax(gt, axis = 1) 
    prior_perpix = weights_2[max_idx_perpix.cpu()]
    prior_perpix = prior_perpix.clone().detach().to(device)

    loss_perpix = prior_perpix * loss_tmp_perpix
    loss = torch.sum(loss_perpix) / (y.shape[2] * y.shape[3])
    
    return loss

def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)

def decode_soft_encoding_batch_ab(y_batch, q_ab, epsilon=1e-8, T=0.38):
    """
    q_ab = np.load(os.path.join(module_dir, "data", "pts_in_hull.npy").replace("\\", "/"))
    x_batch: L channel (batchsize, height, width, 1) --> gray 
    y_batch: softencoding of ab channel (batchsize, height, width, nb_q)
    """
    nb_q = q_ab.shape[0]

    assert len(y_batch.shape) == 4, "Invalid y_batch shape (batchsize, height, width, nb_q)"

    y_batch = y_batch.permute(0, 2, 3, 1)
    y_batch_size, y_height, y_width, y_nb_q = y_batch.shape

    assert y_nb_q == nb_q, "y_nb_q is not equal to q_ab"

    torch_batch_reshape = y_batch.reshape(-1, nb_q)
    torch_batch_reshape = torch.exp(torch.log(torch_batch_reshape + epsilon) / T)
    torch_batch_reshape = torch_batch_reshape / torch_batch_reshape.sum(dim=1, keepdim=True)

    q_a = torch.tensor(q_ab[:, 0]).reshape(1, nb_q).to(device).to(torch.float)
    q_b = torch.tensor(q_ab[:, 1]).reshape(1, nb_q).to(device).to(torch.float)

    torch_batch_a = torch_batch_reshape.matmul(q_a.transpose(0, 1)).reshape(y_batch_size, y_height, y_width, 1) + 128
    torch_batch_b = torch_batch_reshape.matmul(q_b.transpose(0, 1)).reshape(y_batch_size, y_height, y_width, 1) + 128

    torch_batch_ab = torch.cat([torch_batch_a, torch_batch_b], dim=3)
    y_batch_ab = torch_batch_ab.permute(0, 3, 1, 2)
    return y_batch_ab

def merge_lab(l_tensor, ab_tensor):
    '''
    l_tensor: torch.Size([n, 1, 256, 256])
    ab_tensor: torch.Size([n, 313, 64, 64])
    return: np.ndarray - (n, 256, 256, 3)
    '''
    ab_tensor = decode_soft_encoding_batch_ab(ab_tensor, q_ab, epsilon, T)                                 # (n, 2, 64, 64)
    ab_tensor_resized = F.interpolate(ab_tensor, size=(256, 256), mode='bilinear', align_corners=False)    # (n, 2, 256, 256)
    img_tensor = torch.cat([l_tensor, ab_tensor_resized], dim=1)                                           # (n, 3, 256, 256)
    img_np = img_tensor.detach().cpu().numpy()
    img_np = np.clip(img_np, a_min=0, a_max=255)
    img_np = img_np.astype(np.uint8).transpose(0, 2, 3, 1)                                                 # (n, 256, 256, 3)
    return img_np
