import os
import xml.etree.ElementTree as ET

# VOC → YOLO 변환 함수
def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

# YOLO 클래스 이름 정의 (index 순서 중요)
classes = ["carrot"]

# name 매핑: VOC의 숫자 '0'을 'carrot'으로 간주
name_map = {
    '0': 'carrot',
    'carrot': 'carrot'
}

# 경로 설정
splits = ['train', 'valid']
base_path = "D:/git/Carrot_Detection_harvesting_equipment/data/Carrot Detection.v1i.voc"

for split in splits:
    image_dir = os.path.join(base_path, split)
    label_dir = os.path.join(base_path, split, 'labels')
    os.makedirs(label_dir, exist_ok=True)

    for file in os.listdir(image_dir):
        if not file.endswith('.xml'):
            continue

        xml_path = os.path.join(image_dir, file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        txt_path = os.path.join(label_dir, file.replace('.xml', '.txt'))
        with open(txt_path, 'w') as out_file:
            for obj in root.iter('object'):
                cls_raw = obj.find('name').text.strip()
                cls = name_map.get(cls_raw, None)

                if cls is None or cls not in classes:
                    continue

                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (
                    float(xmlbox.find('xmin').text),
                    float(xmlbox.find('xmax').text),
                    float(xmlbox.find('ymin').text),
                    float(xmlbox.find('ymax').text)
                )
                bb = convert((w, h), b)
                out_file.write(f"{cls_id} " + " ".join([f"{a:.6f}" for a in bb]) + "\n")
