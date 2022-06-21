import cv2


def draw_identity(suspect_name, label, cv2_im, bbox, args):
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
