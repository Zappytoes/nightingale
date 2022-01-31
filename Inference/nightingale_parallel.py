import logging
from shapely.geometry import Polygon
import pandas as pd
from osgeo.gdal import Open as gdalOpen
import os
import cv2
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf # tensorflow 1.13.1
tf.logging.set_verbosity(tf.logging.ERROR)
logger = logging.getLogger(__name__)

############## define functions ##################
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


def forward_convert(coordinate, with_label=True):
    """
    :param coordinate: format [x_c, y_c, w, h, theta]
    :return: format [x1, y1, x2, y2, x3, y3, x4, y4]
    """

    boxes = []
    if with_label:
        for rect in coordinate:
            box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
            box = np.reshape(box, [-1, ])
            boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7], rect[5]])
    else:
        for rect in coordinate:
            box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
            box = np.reshape(box, [-1, ])
            boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]])

    return np.array(boxes, dtype=np.float32)


def backward_convert(coordinate, with_label=True):
    """
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4, (label)]
    :param with_label: default True
    :return: format [x_c, y_c, w, h, theta, (label)]
    """

    boxes = []
    if with_label:
        for rect in coordinate:
            box = np.int0(rect[:-1])
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            boxes.append([x, y, w, h, theta, rect[-1]])

    else:
        for rect in coordinate:
            box = np.int0(rect)
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            boxes.append([x, y, w, h, theta])

    return np.array(boxes, dtype=np.float32)


def nms_rotate_cpu(boxes, scores, iou_threshold, max_output_size):

    keep = []

    order = scores.argsort()[::-1]
    num = boxes.shape[0]

    suppressed = np.zeros((num), dtype=np.int)

    for _i in range(num):
        ####print('_i ',_i)
        if len(keep) >= max_output_size:
            break

        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
        area_r1 = boxes[i, 2] * boxes[i, 3]
        for _j in range(_i + 1, num):
            j = order[_j]
            if suppressed[i] == 1:
                continue
            r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
            area_r2 = boxes[j, 2] * boxes[j, 3]
            inter = 0.0

            try:
                int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]

                if int_pts is not None:
                    order_pts = cv2.convexHull(int_pts, returnPoints=True)

                    int_area = cv2.contourArea(order_pts)

                    inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + 1e-5)

            except:
                """
                  cv2.error: /io/opencv/modules/imgproc/src/intersection.cpp:247:
                  error: (-215) intersection.size() <= 8 in function rotatedRectangleIntersection
                """
                # print(r1)
                # print(r2)
                inter = 0.9999

            if inter >= iou_threshold:
                suppressed[j] = 1

    return np.array(keep, np.int64)

def listcol_to_cols(df, new_cols, old_col, inplace=True):
    if not inplace:
        df = df.copy()
    df[new_cols] = df.apply(lambda x: x[old_col], axis=1, result_type="expand")
    df.drop(columns=old_col, inplace=True)
    return None if inplace else df

########### end funcs ##########################################################################

############ class ###############################################

class Detector():
    
    def __init__(self,gpu_ids,placeholder=(2048,2048,3),allow_growth=False,model='model/omitted_scrdet_Frozen.pb'):
        '''
        Assign GPU IDs and Initialize Model Weights
        
        Parameters
        ----------
        gpu_ids : int, [int] or csv string assigning specific GPUs for this process.
                    Current version supports multiple GPUs for parallel processing
                    of large images. 
                    
        placeholder : 3D tuple (rows,cols,channels) sets sliding window size used to
                        process arbitrarily sized imagery and reserves memory on the
                        GPU's. Imagery smaller than placeholder will be zero-padded. 
                        Default is (2048,2048,3). Has been successfully tested up to
                        shape of (4096,4096,3) on a Tesla V100-DGXS-32GB GPU. 
                        
        allow_growth : bool. Whether or not to allow other processes to allocate GPU memory
                        on the GPU's you are using. Default is False. If you are maxing
                        out GPU memeory with very large images (e.g., 4096,4096,3), you
                        will want this set to False. For futher info, see Tensorflow's
                        documentation for tf.ConfigProto gpu_options.allow_growth
                        
        model : string, path to the tensorflow frozen graph .pb file
        '''
        ###
        gpu_devices = str(gpu_ids).replace('[','').replace(']','').replace('(','').replace(')','')
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
        num_devices = len(gpu_devices.split(','))
        #text = 'Initializing detector and warming up '+str(num_devices)+' GPU(s)...'
        #print(text)
        logger.info('Initializing detector and warming up '+str(num_devices)+' GPU(s)...')
        ###
        frozen_graph_file = model
        with tf.gfile.GFile(frozen_graph_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            
        # list variables
        img_g = []
        dets = []

        # Parse the graph_def file
        with tf.gfile.GFile(frozen_graph_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Load the graph_def in the default graph,
        # then add the graph to each available gpu
        # with a unique name based on the device
        feed_dict = {}
        with tf.Graph().as_default() as graph:
            for device in range(0,num_devices):
                with tf.device('/device:GPU:'+str(device)):
                    tf.import_graph_def(graph_def,
                                        input_map=None,
                                        return_elements=None,
                                        name=str(device),
                                        op_dict=None,
                                        producer_op_list=None)
                    
                    img_g.append(graph.get_tensor_by_name(str(device)+"/input_img:0"))
                    dets.append(graph.get_tensor_by_name(str(device)+"/DetResults:0"))
                    feed_dict[img_g[device]] = np.zeros(placeholder,dtype=np.uint8)
                    
        config = tf.ConfigProto(allow_soft_placement=True)
        if allow_growth:
            config.gpu_options.allow_growth = True
        sess=tf.Session(config=config,graph=graph)
        self.dets = dets
        self.img_g = img_g
        self.feed_dict = feed_dict
        sess.run(self.dets, feed_dict=self.feed_dict)
        self.detector = sess
        self.placeholder = placeholder
        self.num_devices = num_devices
        #print('Initialization complete. Ready to run predict.')
        logger.info('Initialization complete. Ready to run predict.')
        
        
    def predict(self,file,clips = None, conf = 0.01,
                class_list = False,
                virtual_mem=False, nms_iou = 0.2, h_overlap = 200,
                w_overlap = 200, max_nms_output = 200):
        
        '''
        Parameters
        ----------
        file : str
            Complete path to an image, or path to a directory of images
        
        clips : list[[y0_0,x0_0,y1_0,x1_0],...[y0_n,x0_n,y1_n,x1_n]]
            A list of lists containing pixel values used to clip out a portion or portions of the image for 
            processing. Can be left as  clips = None to process the whole image (this is 
            default). Must be formatted as a list of list, so passing a single clip would be formatted as 
            clips = [[row0,col0,row1,col1]]. Clips are not supported if passing in a directory of 
            images for processing. Pass a single image with clips locations if you want to use clips. If clips are
            passed in, then virtual_mem will automatically be disabled (set to False).
        
        conf : float (0,1] or list of floats where each value corresponds to a specific class confidence
            Output detection if detection confidence score is greater than or equal to this value
            
        class_list : list of strings
            A name for each class, not including background. Order should match the category integer value set in your label dictionary during model training. E.g., ['car','plane','boat']
        
        virtual_mem : Bool
            Reduce memory consumption by treating NITF imagery as a virtual array, where smaller portions of the 
            NITF image are read into memory as they are needed for processing. Reduces time up front by 
            preventing the whole NITF being read into memory, but results in longer processing time overall. 
            If clips are passed in, then virtual_mem setting is overwritten and set to False.
            
        nms_iou : float; Non-Max Supression IoU threshold.

        *_overlap : int; h (height/vertical) or w (width/horizontal) pixel overlap for tiled/sliding image 
            processing window(s). Size of processing window is set by the "placeholder" parameter
            when the detector is initialized.

        max_nms_output : int; maximum number of detections to return from nms function. 

        
        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the following colums:
                * id - int, detection id (index of dataframe)
                * geometry - Shapely formatted polygon in NITF pixel coordinates
                * class - string, object class
                * conf - float in (0,1]
                * image_name - string
                * (class) - list of floats, confidence score for each class, including background
        '''
        if not class_list:
            logger.error("Error: Set class_list. class_list should not include background. E.g., ['car','plane','boat']")
                return
            
        
        h_len = self.placeholder[0]
        w_len = self.placeholder[1]
        # If only a single value for min_score and nms_iou were input, 
        # turn them into list of repeating values (same value for each class)
        if type(conf) == float:
            conf = [conf]*len(class_list)
        elif type(conf) == list:
            if not len(conf) == len(class_list):
                #print('Error: conf (i.e., min score) must be a single float value or a list float values same length as class_list')
                logger.error('Error: conf (i.e., min score) must be a single float value or a list float values same length as class_list')
                return
        else:
            #print('Error: conf (i.e., min score) must be a single float value or a list float values same length as class_list')
            logger.error('Error: conf (i.e., min score) must be a single float value or a list float values same length as class_list')
            return

        if type(nms_iou) == float:
            nms_iou = [nms_iou]*len(class_list)
        elif type(nms_iou) == list:
            if not len(nms_iou) == len(class_list):
                #print('Error: nms_iou must be a single float value or a list float values same length as class_list')
                logger.error('Error: nms_iou must be a single float value or a list float values same length as class_list')
                return
        else:
            #print('Error: nms_iou must be a single float value or a list float values same length as class_list')
            logger.error('Error: nms_iou must be a single float value or a list float values same length as class_list')
            return
            
            
        detection_dataframe = pd.DataFrame()
        
        # check if it's a single image or a directory of images
        if os.path.isfile(file):
            #print('Runnning a single image')
            logger.info('Runnning a single image')
            images_list = [file]
        elif os.path.isdir(file):
            #print('Running a directory of images (clips not supported when passing a directory)')
            logger.info('Running a directory of images (clips not supported when passing a directory)')
            images_list = []
            for name in os.listdir(file):
                images_list.append(file+'/'+name)   
        else:
            #print('Path Error: ', file,' Not a valid file or directory.')
            logger.error('Path Error: '+file+' Not a valid file or directory.')
            return
            
        len_images_list=len(images_list)
        
        loop_count = 0
        for img_path in images_list:
            
            # clips is only supported when a single image is passed
            if os.path.isdir(file):
                #clips = []
                clips = None
            
            imid, file_ext = img_path.split('/')[-1].split('.')[0], img_path.split('/')[-1].split('.')[-1]
            
            # image read and pre-process
            #text = 'Reading image from disk...'
            clip_list = []
            #print(text,end="\r")
            logger.info('Reading image from disk...')
            if file_ext in ['png','PNG','.png','.PNG']:
                #if not clips:
                if clips == None:
                    clip_list.append(cv2.imread(img_path))
                    #clips = [[0,clip_list[-1].shape[0],clip_list[-1].shape[1],0]]
                    clips = [[0,0,clip_list[-1].shape[0],clip_list[-1].shape[1]]]
                else:
                    junk = cv2.imread(img_path)
                    for clip in clips:
                        #clip_list.append(junk[clip[0]:clip[2],clip[3]:clip[1],:]) #top, right, bottom, left #left1,bottom1,right1,top1
                        clip_list.append(junk[clip[0]:clip[2],clip[1]:clip[3],:])
                        
                
            elif file_ext in ['ntf','.ntf','NTF','.NTF','nitf','.NITF','.nitf','.r0','r0']:
                my_nitf = gdalOpen(img_path)
                bits = int(my_nitf.GetMetadata()['NITF_ABPP'])
                #if not clips:
                if clips == None:
                    if virtual_mem:
                        clip_list.append(my_nitf.GetRasterBand(1).GetVirtualMemArray())
                        #clips = [[0,clip_list[-1].shape[0],clip_list[-1].shape[1],0]]
                        clips = [[0,0,clip_list[-1].shape[0],clip_list[-1].shape[1]]]
                    else:
                        clip_list.append(my_nitf.GetRasterBand(1).ReadAsArray())
                        clip_list[-1] = adjust_image(clip_list[-1],bit_depth_in=bits)
                        clip_list[-1] = np.repeat(clip_list[-1][:,:,np.newaxis],3,axis=2)
                        #clips = [[0,clip_list[-1].shape[0],clip_list[-1].shape[1],0]]
                        clips = [[0,0,clip_list[-1].shape[0],clip_list[-1].shape[1]]]
                else:
                    virtual_mem = False # you can't use virtual mem if using clips
                    for clip in clips:
                        #clip_list.append(my_nitf.GetRasterBand(1).ReadAsArray(clip[0],clip[3],clip[2]-clip[0],clip[1]-clip[3]))
                        clip_list.append(
                            my_nitf.GetRasterBand(1).ReadAsArray(
                                clip[1],
                                clip[0],
                                clip[3]-clip[1],
                                clip[2]-clip[0]))

                        clip_list[-1] = adjust_image(clip_list[-1],bit_depth_in=bits)
                        clip_list[-1] = np.repeat(clip_list[-1][:,:,np.newaxis],3,axis=2)
                
            else:
                logger.warning('Image Format '+file_ext+' not tested yet, skipping')
                #print('Image Format '+file_ext+' not tested yet, skipping')
                loop_count = loop_count + 1
                continue
                #return
            
            # Can I start a new loop here for each clip?
            if virtual_mem:
                dtype=type(clip_list[0][0,0])
            else:
                dtype=type(clip_list[0][0,0,0])
            clip_count = 0
            num_clips = len(clips)
            for img,clip in zip(clip_list,clips):
                rows, cols = img.shape[0:2]
                # Make the array coordiantes of the image chips (for gpu parrallel) using the current clip 
                chips = []
                x_step = int(w_len-w_overlap)
                y_step = int(h_len-h_overlap)
                for y_start in np.arange(0,rows,y_step,dtype=int):
                    for x_start in np.arange(0,cols,x_step,dtype=int):
                        x_end = x_start+w_len 
                        y_end = y_start+h_len

                        chips.append((y_start,y_end,x_start,x_end))

                # pad the image with zeros to maintain memory space for gpu?
                if not virtual_mem:
                    blank = np.zeros((y_end, x_end,3),dtype=dtype)
                    blank[0:rows,0:cols,:] = img
                    img = blank
                    
                #rows, cols, chan = img.shape

                # the chip placement pattern for multiple gpus
                place = []
                for throw in range(0,len(chips),self.num_devices):
                    if throw > len(chips):
                        place.append(chips[throw:len(chips)])
                    else:
                        place.append(chips[throw:throw+self.num_devices])

                # print when the image reading and pre-processing is done
                #print(' '*len(text),end="\r")

                ### loop over single image here for multi-gpu processing ###
                spin_ct = 0
                image_dets_list = [] # reset the master list for this image
                for p in place:
                    text = 'Image '+str(loop_count+1)+' of '+str(len_images_list)+\
                           ' : site chip '+str(clip_count+1)+' of '+str(num_clips)+\
                           ' : subchip '+str(spin_ct+len(p))+' of '+str(len(chips))
                    #print(text,end="\r")
                    logger.info(text)
                    spin_ct = spin_ct + len(p)

                    diff = self.num_devices - len(p)
                    for full in range(0,len(p)):
                        if not virtual_mem:
                            self.feed_dict[self.img_g[full]] = img[p[full][0]:p[full][1],p[full][2]:p[full][3]]
                        else:
                            # no zero padding, so change where the array ends
                            junk_end_row = p[full][1]
                            junk_end_col = p[full][3]
                            if p[full][1] > img.shape[0]:
                                junk_end_row = img.shape[0]
                            if p[full][3] > img.shape[1]:
                                junk_end_col = img.shape[1]
                            
                            # adjust
                            junk = adjust_image(img[p[full][0]:junk_end_row,p[full][2]:junk_end_col],bit_depth_in=bits)
                            # add channel
                            junk = np.repeat(junk[:,:,np.newaxis],3,axis=2)
                            # finally add to feed dict
                            self.feed_dict[self.img_g[full]] = junk

                    # run inference on each chip in p using all available gpus
                    dets_val_par = self.detector.run(self.dets, feed_dict=self.feed_dict)

                    # delete empty dets
                    if diff > 0:
                        del dets_val_par[-diff:]

                    ## Description of dets_val_par index format for each detection.
                    ## dets_val_par should be a list of length num_gpus.
                    ## Each list entry (results from a single GPU) is an array with the following index format:
                        ## 0 = label (integer from 0:4)
                        ## 1 = confidence values of category
                        ## 2:7 = oriented bouding box format: [x_c, y_c, w, h, theta]
                        ## 7: = # vector of confidence values for each category (background included)

                    #correct offsets from each gpu and append detections from each GPU to master list for this image
                    for index in range(0,len(dets_val_par)):
                        #dets_val_par[index][:,2] += p[index][2]+clip[0] # correct the GPU x offset plus the clip x offset
                        #dets_val_par[index][:,3] += p[index][0]+clip[3] # correct the GPU y offset plus the clip y offset
                        dets_val_par[index][:,2] += p[index][2]+clip[1] # correct the GPU x offset plus the clip x offset
                        dets_val_par[index][:,3] += p[index][0]+clip[0] # correct the GPU y offset plus the clip y offset
                        image_dets_list = image_dets_list+list(dets_val_par[index]) # populate the master list for this image

                    #print(' '*len(text),end="\r")

                    ## completed one run of multi-gpu inference with chips in "p"; move on to next chips in "place"##
                # Convert the list to an array
                image_dets_list = np.array(image_dets_list)    
                ## populate the master lists for ALL images ##
                ## new code
                box_res_rotate=image_dets_list[:, 2:7] # oriented bouding box format: [x_c, y_c, w, h, theta]
                label_res_rotate=image_dets_list[:, 0] # labels are values ranging from 0-3 that can be used to index the class_list
                score_res_rotate=image_dets_list[:, 1] # confidence values of category (non-background)
                score_vec_res_rotate=image_dets_list[:,7:] # vector of confidence values for each category (background included)
                

                # NMS
                keep = []
                for cl in np.arange(0,len(class_list)): # detector outputs lables 1-3, skippping 0 (the background class=0)
                    cls_idx = np.where(label_res_rotate == cl+1)[0] # where the class label is greater than background label (zero)
                    cls_min_score_idx = np.where(score_res_rotate[cls_idx] >= conf[cl])[0] # where the class & min score satisfy 
                    keep_cls = nms_rotate_cpu(boxes=box_res_rotate[cls_idx[cls_min_score_idx]],
                                              scores=score_res_rotate[cls_idx[cls_min_score_idx]],
                                              iou_threshold=nms_iou[cl], max_output_size=max_nms_output)
                    keep = keep + list(cls_idx[cls_min_score_idx[keep_cls]])

                keep = np.sort(np.asarray(keep))
                
                ## in case of empty keep array
                if keep.size == 0:
                    box_res_rotate=box_res_rotate[0:1]
                    label_res_rotate=label_res_rotate[0:1]
                    score_res_rotate=score_res_rotate[0:1]
                    score_vec_res_rotate=score_vec_res_rotate[0:1]
                else:
                    box_res_rotate=box_res_rotate[keep]
                    label_res_rotate=label_res_rotate[keep]
                    score_res_rotate=score_res_rotate[keep]
                    score_vec_res_rotate=score_vec_res_rotate[keep]

                # turn the detection info for this image into a pandas dataframe
                polylist = []
                for index in np.arange(0,len(label_res_rotate)):
                    box = np.array(cv2.boxPoints(((box_res_rotate[index,0], box_res_rotate[index,1]), 
                                                  (box_res_rotate[index,2], box_res_rotate[index,3]), 
                                                  box_res_rotate[index,4])),dtype=np.float32) #.round(decimals=1)
                    
                    polylist.append(Polygon([[box[0,0],box[0,1]],
                                             [box[1,0],box[1,1]],
                                             [box[2,0],box[2,1]],
                                             [box[3,0],box[3,1]]]))

                temp_dataframe = pd.DataFrame()
                temp_dataframe['geometry'] = polylist
                temp_dataframe['class'] = label_res_rotate
                temp_dataframe['conf'] = np.round(score_res_rotate,decimals=2)
                temp_dataframe['image_name'] = [imid]*len(label_res_rotate)
                temp_dataframe['(class)'] = list(np.round(score_vec_res_rotate,decimals=2))

                # append the dataframe for the current image to the master dataframe
                detection_dataframe = detection_dataframe.append(temp_dataframe,ignore_index=True)

                clip_count = clip_count+1
                ###### END single image processing, move to next image #############
            loop_count = loop_count + 1
        #print('Completed Image '+str(loop_count)+' of '+str(len_images_list))
        logger.info('Completed Image '+str(loop_count)+' of '+str(len_images_list))
        
        # add the unique identifier column
        detection_dataframe['id'] = np.arange(0,len(detection_dataframe),dtype=int)
        
        for ii in detection_dataframe['class'].unique():
            detection_dataframe['class'] = detection_dataframe['class'].replace(to_replace=ii,value=class_list[int(ii-1)])

        # make a column in the dataframe for each class score for each detection
        listcol_to_cols(detection_dataframe, ['Background']+class_list, '(class)')
        
        return detection_dataframe