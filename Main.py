import cv2
import numpy as np


def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    """
    @brief      Overlays a transparant PNG onto another image using CV2

    @param      background_img    The background image
    @param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
    @param      x                 x location to place the top-left corner of our overlay
    @param      y                 y location to place the top-left corner of our overlay
    @param      overlay_size      The size to scale our overlay to (tuple), no scaling if None

    @return     Background image with overlay on top
    """

    bg_img = background_img.copy()

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    # Extract the alpha mask of the RGBA image, convert to RGB
    b, g, r, a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b, g, r))

    # Apply some simple filtering to remove edge noise
    mask = cv2.medianBlur(a, 5)

    h, w, _ = overlay_color.shape
    roi = bg_img[y:y + h, x:x + w]

    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))

    # Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)

    # Update the original image with our new ROI
    bg_img[y:y + h, x:x + w] = cv2.add(img1_bg, img2_fg)

    return bg_img


cv2.ocl.setUseOpenCL(False)
umbrella = cv2.imread(r"images\PinClipart.com_sky-background-clipart_21089.png", -1)
snow = cv2.imread('images\snowflake-png-image-600.png', -1)  # -1 loads with transparency
rain = cv2.imread('images\clipart2196106.png', -1)  # -1 loads with transparency
# umbrella = cv2.imread('images\umbrella.png', -1)  # -1 loads with transparency
umbrella_shadow = umbrella[:, :, 0:3]
print(umbrella_shadow.shape)

# rain = cv2.imread('download.jpg')

cap = cv2.VideoCapture(0)

n = 50
list_rain = []
has_fallen = []
list_coords = np.zeros([n, 2])
ret, frame = cap.read()
y_offset = 0
x_offset = 0

for i in range(n):
    r = snow.copy()
    has_fallen.append(False)
    list_rain.append(r)

w, h = 20, 30

for i in range(n):
    x = np.random.randint(frame.shape[1] - w)
    list_coords[i, 0] = 0
    list_coords[i, 1] = x

# read video file
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=10, history=5000)

xOffset = yOffset = 0  # this is for umbrella
width = 300
height = 150
dist_prev = 0

while cap.isOpened:

    # if ret is true than no error with cap.isOpened
    ret, frame = cap.read()
    frame2 = frame.copy()

    # if xOffset + height < frame.shape[0] and yOffset + width < frame.shape[1]:
    frame2 = overlay_transparent(frame, umbrella, xOffset, yOffset, overlay_size=(width, height))

    for i in range(n):  # print rain
        y_offset = int(list_coords[i, 0])
        x_offset = int(list_coords[i, 1])
        frame2 = overlay_transparent(frame2, list_rain[i], x_offset, y_offset, (w, h))
        if has_fallen[i]:
            has_fallen[i] = False
            list_rain[i] = snow.copy()
            list_coords[i, 0] = 0

    if ret:

        # apply background substraction
        fgmask = fgbg.apply(frame)
        temp = np.where(frame2 == np.array([0, 0, 0]))
        a = temp[0]
        b = temp[1]
        fgmask[a, b] = 255

        (contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(n):  # check if it reaches contour
            x_offset = int(list_coords[i, 1])
            if (list_coords[i, 0] + 5) + h < frame.shape[0]:
                flag = False
                for j in range(x_offset, x_offset + w):
                    if fgmask[int(list_coords[i, 0]) + 1 + h, j] == 255:  # forground
                        flag = True
                        break

                if not flag:
                    list_coords[i, 0] += 5
                    y_offset = int(list_coords[i, 0])
                else:
                    list_rain[i] = rain.copy()
                    has_fallen[i] = True
            else:
                list_rain[i] = rain.copy()
                has_fallen[i] = True


        for c in contours:
            if cv2.contourArea(c) < 500:
                continue

            # get bounding box from countour
            # (x, y, w, h) = cv2.boundingRect(c)

            # draw bounding box
            x = int(xOffset + width / 2)
            y = int(yOffset + height / 2)
            # print((x,y))
            dist = cv2.pointPolygonTest(c, (x, y), True)
            if dist > 0:
                a = np.array([cv2.pointPolygonTest(c, (x + 5, y), True),
                              cv2.pointPolygonTest(c, (x, y + 5), True),
                              cv2.pointPolygonTest(c, (x + 5, y + 5), True),
                              cv2.pointPolygonTest(c, (x, y - 5), True),
                              cv2.pointPolygonTest(c, (x - 5, y - 5), True),
                              cv2.pointPolygonTest(c, (x - 5, y), True)])
                index = np.argmax(a)
                if dist < a[index]:
                    if index == 0:
                        if xOffset + width + 5 < frame2.shape[1]:
                            xOffset += 5
                    elif index == 1:
                        if yOffset + height + 5 < frame2.shape[0]:
                            yOffset += 5
                    elif index == 2:
                        if xOffset + width + 5 < frame2.shape[1] and yOffset + height + 5 < frame2.shape[0]:
                            xOffset += 5
                            yOffset += 5
                    elif index == 3:
                        if yOffset - 5 > 0:
                            yOffset -= 5
                    elif index == 4:
                        if yOffset - 5 > 0 and xOffset - 5 > 0:
                            xOffset -= 5
                            yOffset -= 5
                    elif index == 5:
                        if xOffset - 5 > 0:
                            xOffset -= 5


            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawContours(frame, c, -1, (0, 255, 0), 3)
        cv2.imshow('foreground and background', fgmask)
        cv2.imshow('rgb', frame)
        cv2.imshow('frame2', frame2)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
