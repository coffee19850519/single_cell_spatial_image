import os.path as osp
import pickle
import shutil
import tempfile
import os
import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
import pandas as pd
import json
import cv2
from PIL import Image
from sklearn.metrics.cluster import adjusted_rand_score
import shutil
def np2tmp(array, temp_file_name=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.

    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False).name
    np.save(temp_file_name, array)
    return temp_file_name


def single_gpu_test(model,
                    data_loader,
                    output_folder,
                    show=False,
                    out_dir=None,
                    efficient_test=False):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    MI_list = []
    name_list = []  

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

            #calculate MI

            img_name = data['img_metas'][0].data[0][0]['filename']
            name = img_name.split('/', 7 )[-1]
            name_list.append(name)
            image_test = cv2.imread(img_name)
            predict = result[0].astype(np.int32)
            if not os.path.exists(output_folder+'result_temp/'):
                os.makedirs(output_folder+'result_temp/')
            np.savetxt(output_folder+'result_temp/'+name.split('.png')[0]+'.csv', predict, delimiter=',')
            MI = cluster_heterogeneity(image_test, predict, 0)
            MI_list.append(MI)
            print('\n',name)  
            print(MI)
        if show or out_dir:
            # print(out_dir)
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)
            # print(imgs)
            # print(imgs)
            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    MI_result = {
                  'name':name_list,
                  # "ARI":ARI_list,
                  'MI':MI_list,
                  }
    MI_result = pd.DataFrame(MI_result)
    MI_result = MI_result.sort_values(by=['MI'],ascending=False)
    # if not os.path.exists('segmentation/QA_result/'):
    #     os.makedirs('segmentation/QA_result/')

    if len(name_list)>5:
        MI_result_top5 = MI_result[0:5]
        # print(MI_result_top5)
        name = MI_result_top5.iloc[:, 0].values
        for n in name:
            prefix = n.split('.png')[0]
            show = cv2.imread(out_dir+n)
            if not os.path.exists(output_folder+'show/'):
                os.makedirs(output_folder+'show/')
            cv2.imwrite(output_folder+'show/'+n,show)
            if not os.path.exists(output_folder+'result/'):
                os.makedirs(output_folder+'result/')
            shutil.move(output_folder+'result_temp/'+prefix+'.csv', output_folder+'result/'+prefix+'.csv')
        shutil.rmtree(out_dir)
        shutil.rmtree(output_folder+'result_temp/')
        # print(name)
        MI_result_top5.to_csv(output_folder+'MI_rank_result.csv',index=True,header=True)
    else:
        name = MI_result.iloc[:, 0].values
        for n in name:
            prefix = n.split('.png')[0]
            show = cv2.imread(out_dir + n)
            if not os.path.exists(output_folder + 'show/'):
                os.makedirs(output_folder + 'show/')
            cv2.imwrite(output_folder + 'show/' + n, show)
            if not os.path.exists(output_folder + 'result/'):
                os.makedirs(output_folder + 'result/')
            shutil.move(output_folder + 'result_temp/' + prefix + '.csv', output_folder + 'result/' + prefix + '.csv')
        shutil.rmtree(out_dir)
        shutil.rmtree(output_folder + 'result_temp/')
        MI_result.to_csv(output_folder+'MI_rank_result.csv',index=True,header=True)

    top1_name = MI_result.iloc[:, 0].values[0]
    top1_csv_name = output_folder+'result/'+top1_name.split('.png')[0]+'.csv'
    print(top1_csv_name)
    return results, top1_csv_name

def cluster_heterogeneity(image_test, category_map, background_category):
    if len(category_map.shape) > 2:
        category_map = cv2.cvtColor(category_map, cv2.COLOR_BGR2GRAY)

    if len(image_test.shape) > 2:
        image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)
    category_list = np.unique(category_map)


    # 邻接矩阵
    W = np.zeros((len(category_list), len(category_list)),dtype=int)
    for i in range(category_map.shape[0]):
        flag1 = category_map[i][0]
        flag2 = category_map[0][i]
        for j in range(category_map.shape[0]):
            if category_map[i][j] != flag1:  # 按行遍历
                index1 = np.where(category_list == flag1)[0][0]
                # print(np.where(category_list == flag1))
                index2 = np.where(category_list == category_map[i][j])[0][0]
                W[index1][index2] = 1
                W[index2][index1] = 1
                flag1 = category_map[i][j]
            if category_map[j][i] != flag2:  # 按列遍历
                index1 = np.where(category_list == flag2)[0][0]
                index2 = np.where(category_list == category_map[j][i])[0][0]
                W[index1][index2] = 1
                W[index2][index1] = 1
                flag2 = category_map[j][i]
    W = W[1:, 1:]  #
    # print(W)

    category_num = W.shape[0]

    # print(category_map[871])
    # 计算每个cluster平均灰度和全图平均灰度
    num = 0
    gray_list = []
    gray_mean = 0
    for category in category_list:
        pixel_x, pixel_y = np.where(category_map == category)
        if category == background_category:
            num = len(pixel_x)
            continue
        gray = []
        for i in range(len(pixel_x)):
            gray.append(image_test[pixel_x[i], pixel_y[i]])
        gray_value = np.mean(gray)
        # print(gray_value)
        gray_list.append(gray_value)
        gray_mean += gray_value * len(pixel_x)
    gray_mean = gray_mean / (image_test.shape[0] ** 2 - num)

    n = W.shape[0]
    a = 0
    b = 0
    for p in range(n):
        index, = np.where(W[p] == 1)
        for q in range(len(index)):
            a += abs((gray_list[p] - gray_mean) * (gray_list[index[q]] - gray_mean))
        b += (gray_list[p] - gray_mean) ** 2
    MI = n * a / (b * np.sum(W))
    # for i in range(n):
    #     adj_num = np.sum(W[i])
    #     if adj_num == n-1:
    #         MI = 0
    #         break
    return MI



def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results with CPU."""
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results with GPU."""
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
