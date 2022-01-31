# small library to evalute text file output of object detection text files

import os
import numpy as np
from shapely.geometry import Polygon
from shapely import wkt
#import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import pandas as pd

# define intersection over union
def iou(det_bbox,gt_bbox):

  # convert groundtruth coordinates to list of tuples
    pts = []
    for step in np.arange(0,len(gt_bbox),2):
        pts.append(tuple(gt_bbox[step:step+2]))

    gt_poly = Polygon(pts) # create gt polygon
    
   # convert detection box coords to list of tuples
    pts = []
    for step in np.arange(0,len(det_bbox),2):
        pts.append(tuple(det_bbox[step:step+2]))
        
    det_poly = Polygon(pts) # create detection polygon
    
    intexu = det_poly.intersection(gt_poly).area/det_poly.union(gt_poly).area

    return intexu
# End intersection over union

# Evaluation code
def Eval(gt_file, det_file, iou_thresh, score_thresh, class_list):

    # 1) First read the formatted text files and turn the info into lists or numpy arrays
    # 1) read gt file
    gt_content = pd.read_csv(gt_file)
    
    # 2) read detection file
    det_content = pd.read_csv(det_file)
    
    # 3) convert everything to arrays
    gt_im_id = gt_content['IMID'].to_numpy(dtype=str)
    gt_bbox = gt_content[['xLF','yLF','xRF','yRF','xRB','yRB','xLB','yLB']].to_numpy(dtype=np.float64)
    gt_class = gt_content['class'].to_numpy(dtype=str)
    
    det_class = det_content['class']
    det_im_id = det_content['image_name']
    det_score = det_content['conf']
    # Turn the Polygon coordinates into individual columns
    x0 = [];y0 = [];x1= [];y1 = [];x2 = [];y2 = [];x3 = [];y3 = []
    for index, row in det_content.iterrows():
        plygn = wkt.loads(row['geometry'])
        x0.append(plygn.exterior.xy[0][0])
        y0.append(plygn.exterior.xy[1][0])
        x1.append(plygn.exterior.xy[0][1])
        y1.append(plygn.exterior.xy[1][1])
        x2.append(plygn.exterior.xy[0][2])
        y2.append(plygn.exterior.xy[1][2])
        x3.append(plygn.exterior.xy[0][3])
        y3.append(plygn.exterior.xy[1][3])
    det_bbox = np.array([x0,y0,x1,y1,x2,y2,x3,y3],dtype=np.float64).T
    
    # create master copies of the groundtruth
    gt_bbox_master = np.copy(gt_bbox)
    gt_im_id_master = np.copy(gt_im_id)
    gt_class_master = np.copy(gt_class)
    
    print('converted txt files to arrays')
    ### end turning csv file info into numpy arrays or lists
    
    # 2)  do the eval with confidence score thresholding

    confm = np.zeros((len(class_list)+1,len(class_list),len(score_thresh))) # confusion matrix 
    tp = np.zeros((len(score_thresh),len(class_list))) # true positives matrix
    fp = np.zeros((len(score_thresh),len(class_list))) # false positives matrix
    fn_1d = np.zeros(len(class_list)) # false negatives vector
    # Count the possible false positives (we begain assuming all 
    # groundtruth will be missed (i.e., false negative) and then prove otherwise during the eval)
    for entry in np.arange(0,len(gt_class_master)):
      index = np.where(np.array(class_list) == gt_class_master[entry])[0][0]
      fn_1d[index] = fn_1d[index] + 1
    fn = np.ones((len(score_thresh),len(class_list)))*fn_1d # convert the vector to a matrix
    
    # test for each score threshold
    #for st in tqdm_notebook(np.arange(0,len(score_thresh))): # the tqdm is just for the completion bar
    for st in np.arange(0,len(score_thresh)): # the tqdm is just for the completion bar
      #print(st)
      # reset the FP lists
      FP_label_list = []
      FP_box_list = []
      FP_IMID_list = []
    
      # reset the groundtruth arrays
      gt_bbox = np.copy(gt_bbox_master)
      gt_im_id = np.copy(gt_im_id_master)
      gt_class = np.copy(gt_class_master)
    
      # Loop over each detection and determine if it's a tp or fp
      #pbar = tqdm_notebook(np.arange(0,len(det_content)))
      for det_num in np.arange(0,len(det_content)):
    
        # track the index of the class (0-17)
        class_index = np.where(np.array(class_list) == det_class[det_num])[0][0]
    
        # first, do the score thresholding
        # print('score=',det_score[det_num])
        if np.float(det_score[det_num]) < score_thresh[st]: # if this detection score is less than the 
                                                            # current score threshold, we ignore it
          pass
    
        else: # otherwise, determine if this detection is a TP or FP
    
          # # track the index of the class (0-17)
          # class_index = np.where(np.array(class_list) == det_class[det_num])[0][0]
          #print(det_class[det_num], 'is position', class_index)
    
          # 1) find where the detection image id matches the gt image id's
          ind_im_id = np.where(gt_im_id == det_im_id[det_num])[0] # these are the indices where the image_id's match
    
          if len(ind_im_id) == 0: # if there are no image id matches in the groundtruth, you have either already used all the groundtruth, or you have images id's in your detection data that are not in your groundtruth labels. A detection at this stage must be a mislabled "background"
            #print('no image match for det num ',det_num)
            #print('WARNING!!!!!!!! You have detections on images that are not in the test set!!!!')
            fp[st,class_index] = fp[st,class_index] + 1
            confm[0,class_index,st] = confm[0,class_index,st] + 1 # add 1 to the background target for this selected class and score thresh
            
            FP_IMID_list.append(det_im_id[det_num])
            FP_label_list.append('FALSE_'+det_class[det_num])
            FP_box_list.append(det_bbox[det_num])
            
    
          else: # if there are image id matches (like their should be)...
            #print('det#',det_num,' matching image found')
            #print(ind_im_id)
            #print(gt_im_id[ind_im_id])
    
            # 2) find indices where the indices of matching image id's also match the class
            ind_of_ind = np.where(gt_class[ind_im_id] == det_class[det_num])[0] # these are the "indices-of-the-indices" that match the class
    
            if len(ind_of_ind) == 0: # if the detection class doesn't match any classes in the image, it's a false positive
              #print('no class match for det num ',det_num)
              fp[st,class_index] = fp[st,class_index] + 1 # increase the false positive count by 1 at the corresponding score-thresh and class element 
              FP_IMID_list.append(det_im_id[det_num])
              FP_label_list.append('FALSE_'+det_class[det_num])
              FP_box_list.append(det_bbox[det_num])
               
              # here we can figure out what the detection was misidentified as by finding the gt with the highest iou with this detection
              my_ious = []
              # 3) find the IOU's for all the NON-matching groundtruths
              for my_image_class_bboxes in gt_bbox[ind_im_id]: # loop over each possible TP and find the IOU
                my_ious.append(iou(det_bbox[det_num],my_image_class_bboxes)) # append the IOU vlaue to this list
    
              my_ious = np.asarray(my_ious) # convert the iou's to an array
              max_iou_index = np.argmax(my_ious) # find the index of the max IOU (think of this as the "max iou index" of the "matching image indices" of the original list")
                     
              
                     
              max_iou = my_ious[max_iou_index] # the value of the max IOU
              #print('Detection #',det_num,' iou=',max_iou)
    
              if max_iou < iou_thresh: # if the max IOU doesn't pass the IOU threshold, it's background 
                #fp[st,class_index] = fp[st,class_index] + 1
                #pass
                confm[0,class_index,st] = confm[0,class_index,st] + 1
              
              else: # max_iou >= iou_thresh: # if the max IOU passess the IOU threshold, then add one to the appropriate target class, detection class, and score threshold
                #tp[st,class_index] = tp[st,class_index] + 1 # award TP
                #fn[st,class_index] = fn[st,class_index] - 1 # remove a FN
                target_class = gt_class[ind_im_id[max_iou_index]]
                target_class_index = np.where(np.array(class_list) == target_class)[0][0]
                confm[target_class_index+1,class_index,st] = confm[target_class_index+1,class_index,st] + 1
              
              
    
            else: # if the detection image_id and class match classes on this image, find the intersection over union for each potential TP
              #print('matching image & class found for det#',det_num,' searching for acceptable IOU')
              #print(ind_of_ind)
              #print(gt_im_id[ind_im_id[ind_of_ind]])
              #print(gt_class[ind_im_id[ind_of_ind]])
    
              my_ious = []
              # 3) find the IOU's for all the matching groundtruths
              for my_image_class_bboxes in gt_bbox[ind_im_id[ind_of_ind]]: # loop over each possible TP and find the IOU
                my_ious.append(iou(det_bbox[det_num],my_image_class_bboxes)) # append the IOU vlaue to this list
    
              my_ious = np.asarray(my_ious) # convert the iou's to an array
              max_iou_index = np.argmax(my_ious) # find the index of the max IOU (think of this as the "max iou index" of the "matching class indices" of the "matching image indices" of the original list")
              max_iou = my_ious[max_iou_index] # the value of the max IOU
              #print('Detection #',det_num,' iou=',max_iou)
    
              if max_iou < iou_thresh: # if the max IOU doesn't pass the IOU 
                                        # threshold, it's a false positive and should be labeled as background
                fp[st,class_index] = fp[st,class_index] + 1
                confm[0,class_index,st] = confm[0,class_index,st] + 1
                
                FP_IMID_list.append(det_im_id[det_num])
                FP_label_list.append('FALSE_'+det_class[det_num])
                FP_box_list.append(det_bbox[det_num])
              
              else: # max_iou >= iou_thresh: # if the max IOU passess the IOU threshold, it's a true positive (ie., this detection is on the same image, with the same class, with an acceptable iou) and the target should match the selection
                tp[st,class_index] = tp[st,class_index] + 1 # award TP
                fn[st,class_index] = fn[st,class_index] - 1 # remove a FN
                     
                target_class = gt_class[ind_im_id[ind_of_ind[max_iou_index]]]
                target_class_index = np.where(np.array(class_list) == target_class)[0][0]
                if target_class_index == class_index:
                     pass
                else:
                     print('target class index is ', target_class_index, 'detection class is ', class_index)
                #confm[target_class+1,class_index,st] = confm[target_class+1,class_index,st] + 1
                confm[target_class_index+1,class_index,st] = confm[target_class_index+1,class_index,st] + 1
    
                # Remove this GT index from the GT list so it can't be used again in the evaluation (at the current score threshold)
                orig_index = ind_im_id[ind_of_ind[max_iou_index]] # get back to the original index of the list
                gt_bbox = np.delete(gt_bbox,orig_index,axis=0)
                gt_im_id = np.delete(gt_im_id,orig_index,axis=0)
                gt_class = np.delete(gt_class,orig_index,axis=0)
        #pbar.update()
      # write the FP list to a csv  
      FP_df = pd.DataFrame(np.array(FP_box_list),columns=['xLF', 'yLF', 'xRF', 'yRF', 'xRB', 'yRB', 'xLB', 'yLB'])
      FP_df['IMID'] = FP_IMID_list
      FP_df['class'] = FP_label_list
      FP_df = FP_df[['IMID', 'xLF', 'yLF', 'xRF', 'yRF', 'xRB', 'yRB', 'xLB', 'yLB','class']]
      
      save_dir = 'FP_Files'
      if not os.path.exists(save_dir):
        os.makedirs(save_dir)
      save_path = os.path.join(save_dir, 'False_Positives_score_'+str(score_thresh[st])+'_and_up.csv')
      FP_df.to_csv(save_path,index=False)
      
        #pbar.update()
    #pbar.close()
    # do the precision, recall and F1 after all the TP, FP, and FN have been accounted for
    print('')
    print('done, computing p r and f1')
    # initialize the matrices as NANs incase there are results where the metrics can't be computed
    p=np.zeros((len(score_thresh),len(class_list)))*np.nan
    r=np.zeros((len(score_thresh),len(class_list)))*np.nan
    f1=np.zeros((len(score_thresh),len(class_list)))*np.nan
    for st in np.arange(0,len(score_thresh)): # for each score threshold row
      for class_num in np.arange(0,len(class_list)): # for each class column
        
        if (tp[st,class_num]+fp[st,class_num]) == 0: # check if the precision demominator will be zero
          pass
        else:
          p[st,class_num] = tp[st,class_num]/(tp[st,class_num]+fp[st,class_num]) # precision
    
        if (tp[st,class_num]+fn[st,class_num]) == 0: # check if the recall denominator will be zero
          pass
        else:
          r[st,class_num] = tp[st,class_num]/(tp[st,class_num]+fn[st,class_num]) # recall
    
        if np.isnan(p[st,class_num]+r[st,class_num]): # if either P or R are nan
          pass
        elif (p[st,class_num]+r[st,class_num]) == 0: # if the F1 denominator is zero
          pass
        else:
          f1[st,class_num] = 2*(p[st,class_num]*r[st,class_num])/(p[st,class_num]+r[st,class_num]) # f1
    
    # find the average metrics for each score threshold
    #p_mean = np.nanmean(p,axis=1)
    #r_mean = np.nanmean(r,axis=1)
    #f1_mean = (2*p_mean*r_mean)/(p_mean+r_mean)
    #f1_mean = np.nanmean(f1,axis=1)
    #tp_sum = np.sum(tp,axis=1)
    #fp_sum = np.sum(fp,axis=1)
    #fn_sum = np.sum(fn,axis=1)
    print('done')
    
    return p,r,f1,tp,fp,fn,confm