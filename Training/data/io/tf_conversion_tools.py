import numpy as np
import copy
import random
import tensorflow as tf

def adjust_image(arr,bit_depth_in=11,bit_depth_out=8):
    '''
    Take an 11-bit image and turn it into an 8-bit one with
    gamma and dynamic range adjustments
    '''
    bscale_in = float(pow(2.0,bit_depth_in))

    bscale_out = int(pow(2.0,bit_depth_out) - 1.0)

    arr = pow(arr /bscale_in, 1/2.2) * bscale_out

    hist,bins = np.histogram(arr.flatten(),bscale_out, [0,bscale_out])
    cdf = hist.cumsum()
    clipL = np.argmax((hist.cumsum()/float(cdf.max()))>.01)
    clipU = np.argmax((hist.cumsum()/float(cdf.max()))>.99)

    #arrAdj = np.clip(arr, clipL, clipU)
    #arrAdj = (arrAdj - arrAdj.min())/(arrAdj.max()-arrAdj.min()) * bscale_out
    
    arr = np.clip(arr, clipL, clipU)
    arr = (arr - arr.min())/(arr.max()-arr.min()) * bscale_out

    if bit_depth_out == 8:
        out_type = 'uint8'
    else:
        out_type = 'uint16'

    #return arrAdj.astype(out_type)
    return arr.astype(out_type)

def format_image_label(gt_dataframe,img_id,label_dict):
    '''
    Maps category string lables to integers and converts categories 
    that are not in the NAME_LABEL_MAP dictionary to background by assigning 
    with integer 0 (back_ground key value must be 0 in NAME_LABEL_MAP). This 
    re-assignement of categories to "background" is useful for retraining a model 
    once False Positives are identified and added to the groundtruth. 
    
    paramters
    -----------------
    gt_dataframe: pandas dataframe: Nightingale formated groundtruth dataframe
    img_id: str: The image id
    label_dict: dict: dictionary of category keys with integer values
    
    returns
    -----------------
    A numpy array of object corner points and integer category values for a given image id
    '''
    format_data = gt_dataframe[gt_dataframe['IMID'] == img_id]
    format_data = format_data[['xLF','yLF','xRF','yRF','xRB', 'yRB','xLB', 'yLB','class']].to_numpy()
    for row in range(0,len(format_data)):
        if format_data[row,8] not in list(label_dict.keys()):
            print('warning, ', format_data[row,8], 'not in NAME_LABEL_MAP. Converting label to background.')
            format_data[row,8] = int(0)
        else:
            #format_data[row,8] = class_list.index(format_data[row,8])
            format_data[row,8] = label_dict[format_data[row,8]]
    
    return np.array(format_data,dtype=np.int32)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def data_2_tfrec(file_idx, image, img_format, boxes_all, width, height, stride_w, stride_h, writer,
               add_random_empties = False, empties_rate_1inX = 150, bits = None):
    '''
    description
    -----------
    Converts a NITF or PNG with groundtruth data (formatted by "format_image_label" function) 
    to a Tensorflow Record for training. Large images will be chipped to smaller sizes of shape 
    heightXwidth to prevent crashing the GPU. Empty image chips (no annotations found on 
    the chip) are by default ingnored, but can be added by setting the 'add_random_empties' 
    and 'empties_rate_1inX' parameters.
    
    If any background category lables (value=0) are found the groundtruth file, they will be 
    removed (background should not be labled for objet detection), but the image chip that 
    contained the background annotations will still be added to the tf_record. This is useful 
    for training because a model can learn to ignore potential false positive objects, so long 
    as they are included in training imagery. 
    
    parameters
    ----------
    file_idx : string : image name without file extension
    
    image : numpy array from cv2.imread() or virtual array from gdal's GetVirtualMemArray
    
    img_format: 'nitf' or 'png'
    
    boxes_all: numpy array of all object annotations (box corners & integer label) for a given image. shape of (num_boxes, 9)
    
    width : int : the desired chip width size to process during training
    
    height: int : the desired chip height size to process during training
    
    stride_w: int : the horizontal stride for chipping the image into smaller sizes
    
    stride_h : int: the vertical stride for chipping the image into smaller sizes
    
    writer: the tf.python_io.TFRecordWriter() object for writing the tensorflow record
    
    add_random_empties : bool : True or False: Whether or not to randomly include empty image chips 
                         in the tf_record (does not affect chips with background lables).
    
    empties_rate_1inX : int : Ignored if add_random_empties=False. 
                        Sets the odds that an empty chip will be added to the
                        training record. E.g., a value of 150 means an empty chip
                        has a 1 in 150 chance that it will be added to the training 
                        record. A value of 1 means an empty chip has a 1 in 1 odds
                        of being added to the training record (i.e., the entire image
                        will be added to the training tf_record). 
                        
    bits: For NITFs only, ignored otherwise. Obtained from gdal's GetMetadata()['NITF_ABPP']
    
    returns
    ----------
    full_count: int: count of chips with annotations that were added to the tf record
    empty_count: int: count of background-only chips that were added to the tf record 
                      (either added from chips that had background annotations in the 
                      groundtruth or annotation-free chips added when add_random_empties=True) 
    
    '''
    pix_mean = []
    pix_mean_normed = []
    pix_std_normed = []
    
    if len(boxes_all) > 0:
        shape = image.shape
        full_count = 0
        empty_count = 0
        for start_h in range(0, shape[0], stride_h):
            for start_w in range(0, shape[1], stride_w):
                boxes = copy.deepcopy(boxes_all)
                box = np.zeros_like(boxes_all)
                start_h_new = start_h
                start_w_new = start_w
                if start_h + height > shape[0]:
                    start_h_new = shape[0] - height
                if start_w + width > shape[1]:
                    start_w_new = shape[1] - width
                top_left_row = max(start_h_new, 0)
                top_left_col = max(start_w_new, 0)
                bottom_right_row = min(start_h + height, shape[0])
                bottom_right_col = min(start_w + width, shape[1])

                subImage = image[top_left_row:bottom_right_row, top_left_col: bottom_right_col]
                
                box[:, 0] = boxes[:, 0] - top_left_col
                box[:, 2] = boxes[:, 2] - top_left_col
                box[:, 4] = boxes[:, 4] - top_left_col
                box[:, 6] = boxes[:, 6] - top_left_col

                box[:, 1] = boxes[:, 1] - top_left_row
                box[:, 3] = boxes[:, 3] - top_left_row
                box[:, 5] = boxes[:, 5] - top_left_row
                box[:, 7] = boxes[:, 7] - top_left_row
                box[:, 8] = boxes[:, 8] # class label
                center_y = 0.25 * (box[:, 1] + box[:, 3] + box[:, 5] + box[:, 7])
                center_x = 0.25 * (box[:, 0] + box[:, 2] + box[:, 4] + box[:, 6])

                cond1 = np.intersect1d(np.where(center_y[:] >= 0)[0], np.where(center_x[:] >= 0)[0])
                cond2 = np.intersect1d(np.where(center_y[:] <= (bottom_right_row - top_left_row))[0],
                                       np.where(center_x[:] <= (bottom_right_col - top_left_col))[0])
                idx = np.intersect1d(cond1, cond2)
                
                to_be_or_not_to_be = np.random.randint(0,empties_rate_1inX)
                
                if subImage.shape[0] > 5 and subImage.shape[1] > 5:
                    if len(idx) > 0: # if there are groundtruth boxes
                        #full_count += 1
                        gtbox_label = box[idx,:]
                        
                        img_name = file_idx+'_'+str(top_left_row)+'_'+str(bottom_right_row)+'_'+ \
                        str(top_left_col)+'_'+str(bottom_right_col)+'.'+img_format
                        
                        #print(img_name)
                        img_height = subImage.shape[0]
                        img_width = subImage.shape[1]
                        
                        if img_format == 'nitf':
                            subImage = adjust_image(subImage,bit_depth_in=bits)
                            subImage = np.repeat(subImage[:,:,np.newaxis],3,axis=2)

                        pix_mean.append([np.mean(subImage[:,:,0]),np.mean(subImage[:,:,1]),np.mean(subImage[:,:,2])])
                        normed_img = subImage / 255
                        pix_mean_normed.append([np.mean(normed_img[:,:,0]),np.mean(normed_img[:,:,1]),np.mean(normed_img[:,:,2])])
                        pix_std_normed.append([np.std(normed_img[:,:,0]),np.std(normed_img[:,:,1]),np.std(normed_img[:,:,2])])
                        #print(pix_mean)
                        #print(pix_mean_normed)
                        #print(pix_std_normed)
                        # Here, remove background lablels before adding to the tf record ############
                        gtbox_label_no_background = []
                        for ii in range(0,gtbox_label.shape[0]):
                            if int(gtbox_label[ii,8]) == 0:
                                print('Removing background label '+str(int(gtbox_label[ii,8]))+' from groundtruth')
                            else:
                                gtbox_label_no_background.append(gtbox_label[ii])

                        if len(gtbox_label_no_background) > 0:
                            gtbox_label = np.array(gtbox_label_no_background,dtype=np.int32)
                            full_count += 1
                        else:
                            gtbox_label = np.array([], dtype=np.int32)
                            empty_count += 1
                        ###########################################################################
                        
                        feature = tf.train.Features(feature={
                                        # do not need encode() in linux
                                        'img_name': _bytes_feature(img_name.encode()),
                                        # 'img_name': _bytes_feature(img_name),
                                        'img_height': _int64_feature(img_height),
                                        'img_width': _int64_feature(img_width),
                                        'img': _bytes_feature(subImage.tostring()),
                                        'gtboxes_and_label': _bytes_feature(gtbox_label.tostring()),
                                        'num_objects': _int64_feature(gtbox_label.shape[0])
                                    })
                        example = tf.train.Example(features=feature)

                        writer.write(example.SerializeToString())
                    
                    elif len(idx) == 0 and add_random_empties and to_be_or_not_to_be == 0:
                        empty_count += 1
                        gtbox_label = np.array([], dtype=np.int32)
                    
                        img_name = file_idx+'_'+str(top_left_row)+'_'+str(bottom_right_row)+'_'+ \
                        str(top_left_col)+'_'+str(bottom_right_col)+'.'+img_format
                        
                        img_height = subImage.shape[0]
                        img_width = subImage.shape[1]
                        
                        if img_format == 'nitf':
                            subImage = adjust_image(subImage,bit_depth_in=bits)
                            subImage = np.repeat(subImage[:,:,np.newaxis],3,axis=2)

                        feature = tf.train.Features(feature={
                                        # do not need encode() in linux
                                        'img_name': _bytes_feature(img_name.encode()),
                                        # 'img_name': _bytes_feature(img_name),
                                        'img_height': _int64_feature(img_height),
                                        'img_width': _int64_feature(img_width),
                                        'img': _bytes_feature(subImage.tostring()),
                                        'gtboxes_and_label': _bytes_feature(gtbox_label.tostring()),
                                        'num_objects': _int64_feature(gtbox_label.shape[0])
                                    })
                        example = tf.train.Example(features=feature)

                        writer.write(example.SerializeToString())
                        
    #print(np.array(pix_mean,dtype=np.float64).shape)
    pix_mean = np.mean(np.array(pix_mean,dtype=np.float64),axis=0)
    pix_mean_normed = np.mean(np.array(pix_mean_normed,dtype=np.float64),axis=0)
    pix_std_normed = np.mean(np.array(pix_std_normed,dtype=np.float64),axis=0)
    pix_metrics = {'pix_mean': pix_mean, 'pix_mean_normed': pix_mean_normed, 'pix_std_normed':pix_std_normed}
    #print(pix_mean)
    #print(pix_mean_normed)
    #print(pix_std_normed)
    return full_count, empty_count, pix_metrics