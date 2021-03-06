import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.config import Args

args = Args()

def cv2ImgAddText(img, text, left, top, textColor, textSize):
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    fontText = ImageFont.truetype(
        args.path_font, textSize, encoding="utf-8")

    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def get_box_info(info_box):
    _, w, _ = info_box.shape
    h_line, _ = get_heights(info_box)
    line_x0 = int(w * args.scale_x)
    line_y0 = int(w * args.scale_y)
    return line_x0, line_y0, h_line

def draw_attr(info_box, text, color, idx_line):
    line_x0, line_y0, h_line = get_box_info(info_box)

    # text
    return cv2ImgAddText(info_box, text, line_x0, line_y0+h_line*idx_line, color, args.textSize)

def draw_identity(info_box, suspect_name, label, color):
    # plot
    _, w, _ = info_box.shape
    line_x0, line_y0, h_line = get_box_info(info_box)

    if suspect_name:
        display_img = args.face_table[suspect_name]
    else:
        display_img = args.face_known
    display_img = cv2.resize(display_img, (w, w))
    info_box[:w, :, :3] = display_img

    # text
    return cv2ImgAddText(info_box, label, line_x0, line_y0+h_line*0, color, args.textSize)

def get_default_info_box(w_new):
    return np.ones(
            (args.scene_height, 1920-w_new, 3),
            dtype=np.uint8
        )*255

def put_default_text(info_box, color = (0, 0, 0), textSize=200):
    h, w, _ = info_box.shape
    h_line = int(h/5)
    x0 = int(w*0.35)
    y0 = int(h*0.01)
    
    info_box = cv2ImgAddText(info_box, '請', x0, y0+h_line*0, color, textSize)
    info_box = cv2ImgAddText(info_box, '靠', x0, y0+h_line*1, color, textSize)
    info_box = cv2ImgAddText(info_box, '近', x0, y0+h_line*2, color, textSize)
    info_box = cv2ImgAddText(info_box, '鏡', x0, y0+h_line*3, color, textSize)
    info_box = cv2ImgAddText(info_box, '頭', x0, y0+h_line*4, color, textSize)
    return info_box

def combine(left, right):
    """
    Stack images horizontally.
    """
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1]
    
    shape = list(left.shape)
    shape[0] = h
    shape[1] = w
    
    comb = np.zeros(tuple(shape), left.dtype)
    
    # left will be on left, aligned top, with right on right
    comb[:left.shape[0],:left.shape[1]] = left
    comb[:right.shape[0],left.shape[1]:] = right
    
    return comb  

def draw_bpm(info_box, crop_bgr, text_bpm, processor, color):
    # plot
    _, w, _ = info_box.shape
    h_line, h_bpm = get_heights(info_box)
    line_x0 = int(w * args.scale_x)
    line_y0 = int(w * args.scale_y)

    plot = make_bpm_plot(processor, crop_bgr, h_bpm, w)
    if plot is not None:
        info_box[-h_bpm:, :, :3] = plot
    
    # text
    info_box = cv2ImgAddText(info_box, text_bpm, line_x0, line_y0+h_line*4, color, args.textSize_bpm)
    return info_box

def make_bpm_plot(processor, crop_bgr, h_bpm=280, w_bpm=640):
    """
    Creates and/or updates the data display
    """
    data = [
            # raw optical intensity
            [processor.times, processor.samples],
            # Power spectral density, with local maxima indicating the heartrate (in beats per minute).
            [processor.freqs, processor.fft]
    ]
    return plotXY(
        data,
        labels=[False, True], showmax=[False, "bpm"], label_ndigits=[0, 0],
        showmax_digits=[0, 1], skip=[3, 3], bg=crop_bgr, size=(h_bpm, w_bpm)
    )

def plotXY(
        data, 
        size = (280,640), margin = 25, 
        labels=[], showmax = [], label_ndigits = [], 
        showmax_digits=[], skip = [], bg = None
    ):
    for x,y in data:
        if len(x) < 2 or len(y) < 2:
            return
    
    n_plots = len(data)
    w = float(size[1])
    h = size[0]/float(n_plots)
    
    z = np.zeros((size[0],size[1],3))
    
    if isinstance(bg, np.ndarray):
        wd = int(bg.shape[1]/bg.shape[0]*h )
        bg = cv2.resize(bg,(wd,int(h)))
        if len(bg.shape) == 3:
            r = combine(bg[:,:,0],z[:,:,0])
            g = combine(bg[:,:,1],z[:,:,1])
            b = combine(bg[:,:,2],z[:,:,2])
        else:
            r = combine(bg,z[:,:,0])
            g = combine(bg,z[:,:,1])
            b = combine(bg,z[:,:,2])
        z = cv2.merge([r,g,b])[:,:-wd,]    
    
    i = 0
    P = []
    for x,y in data:
        x = np.array(x)
        y = -np.array(y)
        
        xx = (w-2*margin)*(x - x.min()) / (x.max() - x.min())+margin
        yy = (h-2*margin)*(y - y.min()) / (y.max() - y.min())+margin + i*h
        if labels:
            if labels[i]:
                for ii in range(len(x)):
                    if ii%skip[i] == 0:
                        col = (255,255,255)
                        ss = '{0:.%sf}' % label_ndigits[i]
                        ss = ss.format(x[ii]) 
                        cv2.putText(z,ss,(int(xx[ii]),int((i+1)*h)),
                                    cv2.FONT_HERSHEY_PLAIN,1,col)           
        if showmax:
            if showmax[i]:
                col = (0,255,0)    
                ii = np.argmax(-y)
                ss = '{0:.%sf} %s' % (showmax_digits[i], showmax[i])
                ss = ss.format(x[ii]) 
                #"%0.0f %s" % (x[ii], showmax[i])
                cv2.putText(z,ss,(int(xx[ii]),int((yy[ii]))),
                            cv2.FONT_HERSHEY_PLAIN,2,col)
        
        try:
            pts = np.array([[x_, y_] for x_, y_ in zip(xx,yy)],np.int32)
            i+=1
            P.append(pts)
        except ValueError:
            pass #temporary

    #hack-y alternative:
    for p in P:
        for i in range(len(p)-1):
            cv2.line(z,tuple(p[i]),tuple(p[i+1]), (255,255,255),1)    
    
    return z

def get_heights(info_box):
    h, w, _ = info_box.shape
    # bpm (280, 640) >> (int(w/640*280), w)
    h_bpm = int(w/640*280)
    h_line = int((h-w-h_bpm)/5)
    return h_line, h_bpm

'''
def draw_identity(suspect_name, label, cv2_im, bbox):
    x0, y0, x1, y1 = bbox
    display_img = cv2.imread(suspect_name)
    display_img = cv2.resize(display_img, (args.pivot_img_size, args.pivot_img_size))
    try:
        resolution_y, resolution_x = cv2_im.shape[:2]
        w = x1-x0
        if y0 - args.pivot_img_size > 0 and x1 + args.pivot_img_size < resolution_x:
            #top right
            cv2_im[y0 - args.pivot_img_size:y0, x1:x1+args.pivot_img_size, :3] = display_img

            overlay = cv2_im.copy(); opacity = 0.4
            cv2.rectangle(cv2_im,(x1,y0),(x1+args.pivot_img_size, y0+20),(46,200,255),cv2.FILLED)
            cv2.addWeighted(overlay, opacity, cv2_im, 1 - opacity, 0, cv2_im)
            cv2.putText(cv2_im, label, (x1, y0+10), args.font, 0.5, args.text_color, 1)

            #connect face and text
            cv2.line(cv2_im,(int((x0+x1)/2), y0), (x0+3*int((x1-x0)/4), y0-int(args.pivot_img_size/2)),(67,67,67),1)
            cv2.line(cv2_im, (x0+3*int((x1-x0)/4), y0-int(args.pivot_img_size/2)), (x1, y0 - int(args.pivot_img_size/2)), (67,67,67),1)

        elif y1 + args.pivot_img_size < resolution_y and x0 - args.pivot_img_size > 0:
            #bottom left
            cv2_im[y1:y1+args.pivot_img_size, x0-args.pivot_img_size:x0, :3] = display_img

            overlay = cv2_im.copy(); opacity = 0.4
            cv2.rectangle(cv2_im,(x0-args.pivot_img_size,y1-20),(x0, y1),(46,200,255),cv2.FILLED)
            cv2.addWeighted(overlay, opacity, cv2_im, 1 - opacity, 0, cv2_im)

            cv2.putText(cv2_im, label, (x0 - args.pivot_img_size, y1-10), args.font, 0.5, args.text_color, 1)

            #connect face and text
            cv2.line(cv2_im,(x0+int(w/2), y1), (x0+int(w/2)-int(w/4), y1+int(args.pivot_img_size/2)),(67,67,67),1)
            cv2.line(cv2_im, (x0+int(w/2)-int(w/4), y1+int(args.pivot_img_size/2)), (x0, y1+int(args.pivot_img_size/2)), (67,67,67),1)

        elif y0 - args.pivot_img_size > 0 and x0 - args.pivot_img_size > 0:
            #top left
            cv2_im[y0-args.pivot_img_size:y0, x0-args.pivot_img_size:x0, :3] = display_img

            overlay = cv2_im.copy(); opacity = 0.4
            cv2.rectangle(cv2_im,(x0 - args.pivot_img_size,y0),(x0, y0+20),(46,200,255),cv2.FILLED)
            cv2.addWeighted(overlay, opacity, cv2_im, 1 - opacity, 0, cv2_im)

            cv2.putText(cv2_im, label, (x0 - args.pivot_img_size, y0+10), args.font, 0.5, args.text_color, 1)

            #connect face and text
            cv2.line(cv2_im,(x0+int(w/2), y0), (x0+int(w/2)-int(w/4), y0-int(args.pivot_img_size/2)),(67,67,67),1)
            cv2.line(cv2_im, (x0+int(w/2)-int(w/4), y0-int(args.pivot_img_size/2)), (x0, y0 - int(args.pivot_img_size/2)), (67,67,67),1)

        elif x1+args.pivot_img_size < resolution_x and y1 + args.pivot_img_size < resolution_y:
            #bottom right
            cv2_im[y1:y1+args.pivot_img_size, x1:x1+args.pivot_img_size, :3] = display_img

            overlay = cv2_im.copy(); opacity = 0.4
            cv2.rectangle(cv2_im,(x1,y1-20),(x1+args.pivot_img_size, y1),(46,200,255),cv2.FILLED)
            cv2.addWeighted(overlay, opacity, cv2_im, 1 - opacity, 0, cv2_im)

            cv2.putText(cv2_im, label, (x1, y1-10), args.font, 0.5, args.text_color, 1)

            #connect face and text
            cv2.line(cv2_im,(x0+int(w/2), y1), (x0+int(w/2)+int(w/4), y1+int(args.pivot_img_size/2)),(67,67,67),1)
            cv2.line(cv2_im, (x0+int(w/2)+int(w/4), y1+int(args.pivot_img_size/2)), (x1, y1+int(args.pivot_img_size/2)), (67,67,67),1)
    except Exception as err:
        print(str(err))
'''
