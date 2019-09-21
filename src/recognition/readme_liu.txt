
3000个epoch之后的模型：
链接:https://pan.baidu.com/s/1m7Tdh07AZZPp7TCEnybUmQ  密码:bmv2

数据集：链接:https://pan.baidu.com/s/1AJoC_oir15MpjpzRWKAm4A  密码:qnkj




1、中文图片识别：
 cd tools
 python recongnize_chinese_pdf.py -c ../data/char_dict/char_dict_cn.json -o ../data/char_dict/ord_map_cn.json --weights_path ../model/crnn_syn90k/shadownet.ckpt --image_path ../data/test_images/test_05.jpg --save_path pdf_recognize_result.txt

2、生成tfrecord文件
（注意字典里面没有的字符需要补充）
按照要求生成训练文件，测试文件和验证文件
/data/test_images/annotation_test.txt
/data/test_images/annotation_train.txt
/data/test_images/annotation_val.txt
/data/test_images/lexicon.txt

3、数据处理：
    执行/Users/liuxiaoyun/PycharmProjects/project/ctpn-with-keras/step12_data_convert.py 生成image和对应的label



4、执行生成tfrecord文件
python write_tfrecords.py -d ../data/test_images -s ../data/tfdata -c ../data/char_dict/char_dict_cn.json -o ../data/char_dict/ord_map_cn.json

5、模型训练：
CPU训练:
 python train_shadownet.py -d ../../../data/recognition -w ../../../model/recognition/crnn_syn90k/shadownet.ckpt -c ../../../data/recognition/char_dict/char_dict_cn.json -o ../../../data/recognition/char_dict/ord_map_cn.json
GPU训练:
python train_shadownet.py -d ../../../data/recognition -w ../../../model/recognition/crnn_syn90k/shadownet.ckpt -c ../../../data/recognition/char_dict/char_dict_cn.json -o ../../../data/recognition/char_dict/ord_map_cn.json -m True

/root/miniconda3/bin/python train_shadownet.py -d ../../../data/recognition -w ../../../model/recognition/crnn_syn90k/shadownet.ckpt -c ../../../data/recognition/char_dict/char_dict_cn.json -o ../../../data/recognition/char_dict/ord_map_cn.json

6、模型测试：
python test_shadownet.py --image_path ../../../data/recognition/image/test/X00016469620-22.jpg --weights_path ../../../model/recognition/crnn_syn90k/shadownet_2019-09-21-17-18-32.ckpt-202000 -c ../../../data/recognition/char_dict/char_dict_cn.json -o ../../../data/recognition/char_dict/ord_map_cn.json -v True


7、大图识别
data/recognition/image/source/X51007846301.jpg
 python recongnize_chinese_pdf.py -c ../../../data/recognition/char_dict/char_dict_cn.json -o ../../../data/recognition/char_dict/ord_map_cn.json --weights_path ../../../model/recognition/crnn_syn90k/shadownet_2019-09-21-17-18-32.ckpt-202000 --image_path ../../../data/recognition/image/source/X51007846301.jpg --save_path ../../../data/recognition/result/X51007846301_result.txt


数据处理的关键性

 python recongnize_chinese_pdf.py -c ../data/char_dict/char_dict_cn.json -o ../data/char_dict/ord_map_cn.json --weights_path model/crnn_syn90k/shadownet_2019-09-20-16-40-04.ckpt-204000 --image_path ../data/test_images/train/X00016469612-0.jpg --save_path pdf_recognize_result.txt