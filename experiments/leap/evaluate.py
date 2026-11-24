import numpy as np
from tools.utils import get_img, cosine_similarity,pic_difference
from settings import settings
import cv2
from matplotlib import pyplot as plt
from settings.settings import video_details
from prettytable import PrettyTable
import json
from collections import Counter
import os

def dict2list(input_dict):
    
    output_list = []
    dict_keys = list(input_dict.keys())
    dict_keys.sort()
    for key in dict_keys:
        output_list.append(input_dict[key])
    return output_list

def DBI_Vec(cluster_dict, similarity_matrix):
    cluster_center = {}
    for label in cluster_dict:
        slaves = cluster_dict[label]
        minimum_dis = 1000000
        minimum_index = 0
        for i in range(len(slaves)):
            slave = slaves[i]
            tem_sum = 0
            for j in range(len(slaves)):
                slave_ = slaves[j]
                dist_value = similarity_matrix[slave][slave_]
                tem_sum += dist_value
            if tem_sum < minimum_dis:
                minimum_dis = tem_sum
                minimum_index = slave
        cluster_center[label] = minimum_index

    S_list = {}
    for label in cluster_dict:
        tem_dist = 0
        for id in cluster_dict[label]:
            tem_dist += similarity_matrix[id][cluster_center[label]]
        S_list[label] = tem_dist / len(cluster_dict[label])

    DBI = 0
    for label1 in cluster_dict:
        max_value = -1
        for label2 in cluster_dict:
            if label1 != label2:
                tem_value = (S_list[label1] + S_list[label2]) / (similarity_matrix[cluster_center[label1]][cluster_center[label2]])
                if tem_value > max_value:
                    max_value = tem_value
        DBI += max_value
    return DBI / len(cluster_dict)


def SI(cluster_dict, similarity_matrix):
    ss = []
    for label in cluster_dict:
        id_list = cluster_dict[label]
        if len(id_list) == 1:
            ss.append(0)
        else:
            for id1 in id_list:
                dist_list = []
                for id2 in id_list:
                    dist_list.append(similarity_matrix[id1][id2])
                a = np.mean(dist_list)
                b = np.inf
                for another_label in cluster_dict:
                    if another_label != label:
                        id_list_another = cluster_dict[another_label]
                        dist_list = []
                        for id3 in id_list_another:
                            dist_list.append(similarity_matrix[id1][id3])
                        b = min(b, np.mean(dist_list))
                ss.append((b - a) / max(a, b))
    return np.mean(ss)

    """
    Query Evaluate
    Selection Query
    Aggregation Query
    Top-K Query
    """
def parse_intervals(resolved_intervals):

    return_tuple = {}
    for apply_id, contents in resolved_intervals.items():
        t_id = -1
        distance = 1e5
        for key,content in contents[1].items():
            dis = content[-1]
            traj_id = content[0]
            if dis < distance:
                distance = dis
                t_id = traj_id
        class_id_list = []
        for detect in contents[0]:
            class_id_list.append(detect[-2])
        class_id = Counter(class_id_list).most_common(2)[0][0]
        if len(Counter(class_id_list).most_common(2))>1:
            
            if 7 in class_id_list:
                class_id = 7
                
        intervals = contents[-1]
        return_tuple[apply_id] = [intervals[0],t_id,class_id]
        
    return return_tuple
        
    
from settings import settings

class evaluation(object):
    def __init__(self,label_type,object_type,cfg):

        self.label_type = label_type
        self.object_type_class = settings.coco_names[object_type]
        self.object_type = object_type
        self.cfg = cfg
        
    def preprocess(self,input_label,input_pred):
        
        self.pred_result = [0 for i in range(self.cfg['end_frame'])]
        self.gt_result = [0 for i in range(self.cfg['end_frame'])]
        self.pred_id = []
        self.gt_id = []
        
        if self.label_type == "pseudo_tuple":

            for key in input_label.keys():
                if input_label[key][-1] == self.object_type_class:
                    if (input_label[key][0] < self.cfg['end_frame']) and (input_label[key][1]\
                        >self.cfg['start_frame']):
                        self.gt_id.append(key)
                        for i in range(max(input_label[key][0],self.cfg['start_frame']),\
                            min(input_label[key][1],self.cfg['end_frame'])):
                            self.gt_result[i] += 1
                    
                    
        elif self.label_type == "pseudo_label":

            for i in range(self.cfg['start_frame'],self.cfg['end_frame']):                
                for label in input_label[i]:                    
                    if label[-1] == self.object_type:
                        if label[0] not in self.gt_id:
                            self.gt_id.append(int(label[0]))
                        self.gt_result[i] += 1
        

        elif self.label_type == "hand_label":

            for key in input_label.keys():
                if input_label[key][0] == self.object_type:
                    if input_label[key][1] < self.cfg['end_frame'] and input_label[key][2] > self.cfg['start_frame']:
                        self.gt_id.append(key)
                        for i in range(max(input_label[key][1],self.cfg['start_frame']),\
                            min(input_label[key][2],self.cfg['end_frame'])):
                            self.gt_result[i] += 1    


        elif self.label_type == "pseudo_origin_tuple":

            for key in input_label.keys():
                if input_label[key][-1] == self.object_type_class:
                    if (input_label[key][0] < self.cfg['end_frame']) and (input_label[key][1]\
                        >self.cfg['start_frame']):
                        self.gt_id.append(key)
                        for i in range(max(input_label[key][0],self.cfg['start_frame']),\
                            min(input_label[key][1],self.cfg['end_frame'])):
                            self.gt_result[i] += 1
        
        else:
            raise ValueError("label_type error")
        
        for key in input_pred.keys():
            if input_pred[key][-1] == self.object_type_class:
                if input_pred[key][0][0]<self.cfg['end_frame'] and input_pred[key][0][1]>self.cfg['start_frame']:
                    self.pred_id.append(key)
                    for i in range(max(input_pred[key][0][0],self.cfg['start_frame']),min(input_pred[key][0][1],self.cfg['end_frame'])):
                        self.pred_result[i] += 1
        
    def selection_query_1(self):
        
        TP,FP,TN,FN = 0,0,0,0
        for frame_id in range(self.cfg['start_frame'], self.cfg['end_frame']):
            
            if self.gt_result[frame_id]>=1 and self.pred_result[frame_id]>=1:
                TP += 1
            elif self.gt_result[frame_id]>=1 and self.pred_result[frame_id]==0:
                FN += 1
            elif self.gt_result[frame_id]==0 and self.pred_result[frame_id]>=1:
                FP += 1
            else: 
                TN += 1
                
        if TP==0 :
            TP=1
                
        recall = round((TP)/(TP+FN),4)
        precision = round((TP)/(TP+FP),4)
        accuracy = round((TP+TN)/(TP+TN+FP+FN),4)

        if TP == 0:
            F1 = 0 
        else:
            F1 = round(2*(precision*recall)/(precision+recall),4)

        return F1, recall, precision, accuracy
    
    def aggregation_query_1(self):
        """
        Query: Count the number of cars per frame
        """
        MAE = 0
        ACC = 0
        for i in range(self.cfg['start_frame'],len(self.pred_result)):
            MAE += abs(self.pred_result[i]-self.gt_result[i])
            if self.gt_result[i]!=0:
                ACC += 1 - (abs(self.pred_result[i]-self.gt_result[i])/self.gt_result[i])
            else:
                if self.pred_result[i]!=0:
                    ACC+=0
                else:
                    ACC+=1
        MAE = MAE/len(self.pred_result)
        ACC = ACC/len(self.pred_result)

        return MAE, ACC
    
    def aggregation_query_2(self,frame_num=10,gap=300):
                
        selected_frame = 0
        hit_count = 0
        last_select = -100
        object_count = self.cfg["%s_count"%(self.object_type)]

        for i in range(self.cfg['start_frame'],self.cfg['end_frame'],10):
            if (self.pred_result[i]>=object_count) and ((i-last_select)>gap):
                selected_frame+=1
                for nei_id in range(max(0,i-45),min(i+45,len(self.pred_result))):
                    if self.gt_result[nei_id]>=object_count:
                        hit_count+=1
                        break
                last_select = i
            if selected_frame == frame_num:
                break

        if frame_num==0:
            frame_num = 1 

        return hit_count/frame_num
            
    def aggregation_query_3(self):
        
        return len(set(self.gt_id)),len(set(self.pred_id))

    def top_k_query_1(self,k=10,gap=300):

        gt_result = np.array(self.gt_result)
        pred_result = np.array(self.pred_result)
        topk_ids = gt_result.argsort()[::-1]
        topk_ids = topk_ids[:5000]
        max_gt = self._sort_max_intervals(topk_ids,gap)
        max_gt.sort()
        max_gt = max_gt[::-1][:k]
        topk_ids = pred_result.argsort()[::-1]
        topk_ids = topk_ids[:5000]
        max_pred = self._sort_max_intervals(topk_ids,gap)
        max_pred.sort()
        max_pred = max_pred[::-1][:k]
        counter_gt = Counter(max_gt)
        counter_pred = Counter(max_pred)

        overlap = sum((counter_gt & counter_pred).values()) / k
        
        return overlap       
        

    def _sort_max_intervals(self, input, gap):

        selected_id = []
        max_set = []

        for id in input:
            if len(selected_id)==0:
                selected_id.append([id,id])
            else:
                match_flag = False
                for clus in selected_id:
                    if id>clus[0] and id<clus[-1]:
                        clus.append(id)
                        clus.sort()
                        match_flag = True
                        break
                    elif (id>clus[-1]) and (id<clus[-1]+int(0.25*gap)):
                        clus.append(id)
                        match_flag = True
                        break
                    elif (id<clus[0]) and (id>clus[0]-int(0.25*gap)):
                        clus.insert(0,id)
                        match_flag = True
                        break
                if not match_flag:
                    selected_id.append([id,id])

        for selected in selected_id:
            tmp = [self.gt_result[i] for i in range(selected[0],selected[-1]+1)]
            max_set.append(max(tmp))
        return max_set
            
        
    
def evaluate_parsed_result(return_tuple,cfg,args,logger):

    label_detailed = False

    if label_detailed:
        car_count = 0
        for key in video_details.gt_tuple.keys():
            if video_details.gt_tuple[key][1]>(cfg["start_frame"] + 50) and video_details.gt_tuple[key][0]<(cfg["end_frame"]-50):
                car_count+=1
        
        car_true_ids = {}
        for frame_num in range(cfg["start_frame"], cfg["end_frame"]):
            gt_detects = video_details.gt_labels[frame_num]
            for detect in gt_detects:
                if int(detect[0]) in car_true_ids.keys():
                    car_true_ids[int(detect[0])] += 1
                else:
                    car_true_ids[int(detect[0])] = 1 
        
        car_count_30 = 0
        for car_id in car_true_ids:
            if car_true_ids[car_id] >=30 :
                car_count_30 += 1
                
        logger.info("MAE: %f" % MAE)
        logger.info("MAPE: %f" % MAPE)
        
        id_recall = []
        id_recall_correct = []
        id_dict = video_details.match_dict
        for key in return_tuple.keys():        
            gt_id = id_dict[key]
            id_recall.append(gt_id)
            if gt_id!=-1 and gt_id in car_true_ids.keys() and car_true_ids[gt_id]>=30:
                id_recall_correct.append(gt_id)


    else:
        car_detected = len(return_tuple.keys())
        car_labeled = len(hand_labeled.keys())
        logger.info("MAE: %f" % MAE)
        logger.info("MAPE: %f" % MAPE)
        
    json_result = json.dumps(return_tuple)

    out_dir = getattr(args, "output_dir", "./outputs")
    os.makedirs(f'{out_dir}/results/parsed', exist_ok=True)
    with open(f"{out_dir}/results/parsed/"+cfg["video_name"]+".json","w") as json_file:
        json_file.write(json_result)
    json_file.close()

    with open(f"{out_dir}/results/parsed/"+cfg["video_name"]+"_object.json","w") as json_type_file:
        object_result = json.dumps(video_details.object_type)
        json_type_file.write(object_result)
    json_type_file.close()
    
    with open(f"{out_dir}/results/parsed/"+cfg["video_name"]+"_match_dict.json","w") as json_dict_file:
        object_result = json.dumps(video_details.match_dict)
        json_dict_file.write(object_result)
    json_dict_file.close()
 
def evaluate_query_result(args,cfg,tuple_origin):
    
    label_types = {"pseudo_tuple":video_details.gt_tuple,"pseudo_label":video_details.gt_labels,\
                   "pseudo_origin_tuple":tuple_origin}
    object_names = ["car","truck","bus"]
    
    for label_type in label_types.keys():
        
        print("===================== %s =====================\n"%(label_type))
        ptabel_q1 = PrettyTable()
        ptabel_q2 = PrettyTable()
        ptabel_q3 = PrettyTable()
        ptabel_q4 = PrettyTable()
        ptabel_q5 = PrettyTable()
        ptabel_q1.field_names = ["Vehicle Class", "F1-Score","Recall","Precision","Accuracy"]
        ptabel_q2.field_names = ["Vehicle Class", "MAE", "ACC"]
        ptabel_q3.field_names = ["Vehicle Class", "Gt Count", "Pred Count"]
        ptabel_q4.field_names = ["Vehicle Class", "Desc", "Accuracy"]
        ptabel_q5.field_names = ["Vehicle Class", "Accuracy"]
        
        for object_name in  object_names:
            evaluate_object = evaluation(label_type, object_name, cfg)
            evaluate_object.preprocess(label_types[label_type],video_details.return_tuple)
            F1,recall,precision,accuracy = evaluate_object.selection_query_1()
            ptabel_q1.add_row([object_name, F1, recall, precision, accuracy])
            MAE,ACC = evaluate_object.aggregation_query_1()
            ptabel_q2.add_row([object_name, MAE, ACC])
            GT_COUNT,PRED_COUNT = evaluate_object.aggregation_query_3()
            ptabel_q3.add_row([object_name, GT_COUNT, PRED_COUNT])
            agg_acc = evaluate_object.aggregation_query_2()
            desc = "Count: %d Gap: 300"%(cfg["%s_count"%(object_name)])
            ptabel_q4.add_row([object_name, desc, agg_acc])
            acc_topk = evaluate_object.top_k_query_1()
            ptabel_q5.add_row([object_name, acc_topk])
            
        print("===================== Query-1  =====================")
        print(ptabel_q1)
        print("===================== Query-1  =====================\n")
        print("===================== Query-2  =====================")
        print(ptabel_q2)
        print("===================== Query-2  =====================\n")
        print("===================== Query-3  =====================")
        print(ptabel_q3)
        print("===================== Query-3  =====================\n")
        print("===================== Query-4  =====================")           
        print(ptabel_q4)
        print("===================== Query-4  =====================\n")  
        print("===================== Query-5  =====================") 
        print(ptabel_q5)
        print("===================== The End  =====================\n")    
        
        
def evaluated_object_recall(frame_sampled,gt_tuple):
    
    item_dict = gt_tuple.keys()
    item_recall = []
    for frame_id in frame_sampled:
        for key in gt_tuple.keys():
            if frame_id >= gt_tuple[key][0] and frame_id <= gt_tuple[key][1]:
                item_recall.append(key)
                
    return len(set(item_recall))/len(item_dict) if len(item_dict) > 0 else 0
    

if __name__ == "__main__":
    pass
