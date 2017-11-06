import caffe
import numpy as np
import math
import cv2
def entry_index(l_w,l_h,location,entry,classes,coords):
    n = location/(l_w*l_h)
    loc = location % (l_w*l_h)
    return n*l_w*l_h*(coords+classes+1)+entry*l_w*l_h+loc
    
def logistic_act(x):
    return 1./(1.+math.exp(-x))


def get_region_box(x,data_bias,n,index,i,j,w,h,stride):
    box = [0,0,0,0]
    box[0] = (i+x[index+0*stride])/w
    box[1] = (j+x[index+1*stride])/h
    box[2] = math.exp(x[index+2*stride])*data_bias[2*n]/w
    box[3] = math.exp(x[index+3*stride])*data_bias[2*n+1]/h
    return box

def activate_array(x,n):
    for i in range(0,n):
        x[i] = logistic_act(x[i])
    

def get_region_boxes(data_in,l_w,l_h,l_n,l_classes,l_bias,thresh,boxes,probs):
    for i in range(0,l_w*l_h):
        row = i / l_w
        col = i % l_w
        for n in range(0,l_n):
            index = n*l_w*l_h+i
            obj_index = entry_index(l_w,l_h,n*l_w*l_h+i,4,0,4)
            box_index = entry_index(l_w,l_h,n*l_w*l_h+i,0,0,4)
            #p_index = index*(l_classes+5)+4
            scale = data_in[obj_index]
            #box_index = index*(l_classes+5)
            boxes[index]=get_region_box(data_in,l_bias,n,box_index,col,row,l_w,l_h,l_w*l_h)
            probs[index]=scale
    
def forward_region_layer(data_in,l_w,l_h,ord_num):
    for n in range(0,ord_num):
        index = entry_index(l_w,l_h,n*l_w*l_h,0,0,4)
        activate_array(data_in[index:],2*l_w*l_h)
        index = entry_index(l_w,l_h,n*l_w*l_h,4,0,4)
        activate_array(data_in[index:],l_w*l_h)
        

def output_detections2(nimgw,nimgh,num,thresh,classes,boxes,probs,pout,max_out):
    nDst = 0
    for i in range(0,num):
        prob = probs[i]
        if prob<thresh:
            continue
        b = boxes[i]
        left = (b[0]-b[2]/2)*nimgw
        right = (b[0]+b[2]/2)*nimgw
        top = (b[1]-b[3]/2)*nimgh
        bot = (b[1]+b[3]/2)*nimgh
        if left<0:
            left=0
        if right>(nimgw-1):
            right = nimgw-1
        if(top<0):
            top=0
        if bot>(nimgh-1):
            bot=nimgh-1
        nCurThres=thresh
        if(bot>=nimgh-2) and (right-left)>1.2*(bot-top):
            nCurThres=thresh*2
            if nCurThres<0.3:
                nCurThres = 0.3
        if (bot-top)>1.2*(right-left) and (left<=2 or right>=(nimgw-2)):
            nCurThres=thresh*2
            if nCurThres<0.3:
                nCurThres = 0.3
        if prob>nCurThres:
            if nDst<max_out:
                pout[nDst] = [left,top,right,bot,prob]
                nDst = nDst + 1
    return nDst
        
    
def demo_forward(proto_path,img_path,model_path,grid_size,coords,thresh):
    #root='/yale/jinw/work2/caffe_cvt/caffe-master/yolo2caffe/person/'   #根目录
    #deploy=root + 'BN0.prototxt'    #deploy文件
    #caffe_model=root + 'BN0.caffe'   #训练好的 caffemodel
    #img=root+'2.jpg'    #随机找的一张待测图片
    deploy = proto_path
    caffe_model = model_path
    img = img_path
    #labels_filename = root + 'mnist/test/labels.txt'  #类别名称文件，将数字标签

    net = caffe.Net(deploy,caffe_model,caffe.TEST)
    #cha kan cai shu
    #for param_name in net.params.keys():
     #   weight = net.params[param_name][0].data
     #   bias = net.params[param_name][1].data
   
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  #设定图片的shape格式(1,3,28,28)
    transformer.set_transpose('data', (2,0,1))    #改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
    #transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))    #减去均值，前面训练模型时没有减均值，这儿就不用
    #transformer.set_raw_scale('data', 255)    # 缩放到【0，255】之间
    #transformer.set_channel_swap('data', (2,1,0))   #交换通道，将图片由RGB变为BGR

    im=caffe.io.load_image(img)                   #加载图片
    net.blobs['data'].data[...] = transformer.preprocess('data',im)      #执行上面设置的图片预处理操作，并将图片载入到blob中

    #nnnnn = net.blobs['data'].data
    
#执行测试
    img_w = int(net.blobs['data'].width)
    img_h = int(net.blobs['data'].height)
    l_w = img_w/int(grid_size)
    l_h = img_h/int(grid_size)
    l_n = int(coords)
    out = net.forward()
    data_in= net.blobs['conv19'].data[0].flatten()
    forward_region_layer(data_in,l_w,l_h,l_n)
    l_bias = [1.4,3.2,1.84,4.3,2.3,3.8,0.98,2.8,0.78,2.2]
    box_arr = [0,0,0,0]
    box_arr2 = [0,0,0,0,0]
    boxs_arr = [box_arr]*10000
    fprobs = [0]*999
    pout = [box_arr2]*10000
    get_region_boxes(data_in,l_w,l_h,l_n,0,l_bias,thresh,boxs_arr,fprobs)
    out_num = output_detections2(img_w,img_h,l_w*l_h*l_n,0.1,0,boxs_arr,fprobs,pout,256)
    im_show = cv2.imread(img)
    print out_num
    for i in range(0,out_num):
        cv2.rectangle(im_show,(int(pout[i][0]),int(pout[i][1])),(int(pout[i][2]),int(pout[i][3])),(255,0,0),2)
    cv2.imshow('123',im_show)
    cv2.waitKey(0)

    

if __name__=='__main__':
    import sys
    if len(sys.argv) < 7:
        print 'need 6 full params'
        sys.exit()
        
    img_path = sys.argv[1]
    proto_path = sys.argv[2]
    model_path = sys.argv[3]
    grid_size = sys.argv[4]
    coords = sys.argv[5]
    thresh = sys.argv[6]
    #grid_size=32,coords=5,thresh=0.1
    demo_forward(proto_path,img_path,model_path,grid_size,coords,thresh)

