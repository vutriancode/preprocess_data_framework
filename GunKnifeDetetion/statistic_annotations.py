import json
import numpy
import pandas
from lxml import etree
import matplotlib
import os
import shutil
import splitfolders
import os
configs = None
with open("/home/anvt/Desktop/GunKnifeDetetion/config.json","rb") as config_file:
    configs = json.load(config_file)


class DataProcess:


    def __init__(self,dataset_name = "Weapons-Image-2fps",yolo_folder_data = ""):
        self.number_of_raw_annotations = 0
        self.annotation_list = {}
        self.dataset_name = dataset_name
        self.xml_path = ""
        self.image_path = ""
        self.image_links = configs["DATASETS"][self.dataset_name]["DATA"]
        self.yolo_folder_data = yolo_folder_data

    def preprocess_data(type):
        def wrapper(func):
            def inner(self,*args, **kwargs):
                for image_link in self.image_links:
                    images_path  = image_link["IMAGE_PATH"]
                    xmls_path =  image_link["ANNOTATION_PATH"]
                    for image_path in os.listdir(images_path):
                        if "." in image_path and not image_path.endswith("xml"):
                            image_extension = image_path.split(".")[-1]
                            xml_path =image_path[:-5] + image_path[-5:].replace(image_extension,"xml")
                            image_full_path = os.path.join(images_path,image_path)
                            annotation_full_path =  os.path.join(xmls_path,xml_path)
                            self.xml_path = annotation_full_path
                            self.image_path = image_full_path
                            if type =="statistic":
                                func(self,*args, **kwargs)

                            elif type == "clean_data":
                                func(self,*args, **kwargs)
                            elif type =="convert_data":
                                func(self,*args, **kwargs)
                if type == "statistic":
                    print(self.annotation_list)

            return inner

        return wrapper
    
    @preprocess_data("statistic")
    def statistic_dataset(self):
        doc  = etree.parse(self.xml_path)
            
        for obj in doc.xpath('//object'):
            name= "fail"
            try:
                name = obj.xpath('./name/text()')[0]
                if name not in self.annotation_list:    
                    self.annotation_list[name] = 1
                else:
                    self.annotation_list[name] += 1
            except:
                pass

    def get_annotation(self,obj):
        name = obj.xpath('./name/text()')[0]
        xmin = int(float(obj.xpath('./bndbox/xmin/text()')[0]))
        ymin = int(float(obj.xpath('./bndbox/ymin/text()')[0]))
        xmax = int(float(obj.xpath('./bndbox/xmax/text()')[0]))
        ymax = int(float(obj.xpath('./bndbox/ymax/text()')[0]))
        return name, xmin, ymin, xmax, ymax
        
    def get_image_size(self,obj):
        obj = obj.xpath('//size')[0]
        width = int(float(obj.xpath('./width/text()')[0]))
        height = int(float(obj.xpath('./height/text()')[0]))
        return width, height
    
    def convert(self,im, box):
        im_width,im_height= im
        _, xmin, ymin, xmax, ymax = box

        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin

        return center_x / im_width, center_y / im_height, width / im_width, height / im_height
    
    @preprocess_data("clean_data")
    def clean_dataset(self):
        doc  = etree.parse(self.xml_path)
        if len(doc.xpath('//object')) == 0:
            os.remove(self.image_path)
            os.remove(self.xml_path)
            
    @preprocess_data("convert_data")
    def convert_xml_to_yolo_data_format(self):
        doc  = etree.parse(self.xml_path)
        width, height = self.get_image_size(doc)
        str_cv = ""
        for obj in doc.xpath('//object'):
            name= "fail"
            try:
                name, xmin, ymin, xmax, ymax = self.get_annotation(obj)
                bbox = [name, xmin, ymin, xmax, ymax]
                i = 0
                is_class = False
                for i in list(configs["DATA_MAP"].keys()):
                    if name.lower() in configs["DATA_MAP"][i]:
                        is_class = True
                        break
                converted_box = self.convert([width, height],bbox)
                if converted_box[2]>0.6 or converted_box[3]>0.6:
                    return
                if is_class:
                    str_cv += str(i) + " " + str(converted_box[0]) + " " + str(converted_box[1]) + " " +str(converted_box[2]) + " " +str(converted_box[3]) + "\n"
            except Exception as e:
                print(e)
                pass
        if str_cv!="":
            label_file_name = self.dataset_name.lower().replace("-","_") + self.xml_path.split("/")[-1].replace("xml","txt").replace("-","_") 
            image_file_name = self.dataset_name.lower().replace("-","_") + self.image_path.split("/")[-1].replace("-","_")
            if not os.path.exists(self.yolo_folder_data +f"/labels/" +label_file_name):
                if not os.path.exists(self.yolo_folder_data +f"/labels/"):
                    os.mkdir(self.yolo_folder_data +f"/labels/")
                if not os.path.exists(self.yolo_folder_data +f"/images/"):
                    os.mkdir(self.yolo_folder_data +f"/images/")
                with open(self.yolo_folder_data +f"/labels/" +label_file_name,"w") as label_file:
                    label_file.write(str_cv)
                    dest = shutil.copyfile(self.image_path, self.yolo_folder_data +f"/images/" +image_file_name)
        return True
    

def shuffle_and_split_dataset():
    path = configs["YOLO_LABELS"]
    splitfolders.ratio(path,seed=1337, output=configs["YOLO_LABELS"] + "/yolov8_data", ratio=(0.7, 0.2, 0.1))
shuffle_and_split_dataset()

# for i in list(configs["DATASETS"].keys()):
#     a = DataProcess(i,configs["YOLO_LABELS"])
#     a.statistic_dataset()
#     a.clean_dataset()
#     a.convert_xml_to_yolo_data_format()
