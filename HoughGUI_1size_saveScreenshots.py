# Copyright 2026 Trevor Gerdes. All rights reserved.
# This software is licensed under the MIT License.

import os
import tkinter as tk
import tkinter.font as font
from tkinter import *
from tkinter import filedialog as fd
from tkinter import ttk
from tkinter import messagebox

import cv2 as cv
import numpy as np


def main():
    HoughTestGUI_window()


def HoughTestGUI_window():
    global root1
    root1 = tk.Tk()
    root1.geometry("1200x400")
    root1.resizable(True, True)
    root1.title('Hough Circle Transform Parameter Tester GUI')

    buttonsFont = font.Font(size=14, weight='bold')

    button_HoughTestGUI_0 = tk.Button(root1, text='STEP 1: Select Tester Image', command=HoughTestGUI_selectTesterImage,
                                      font=buttonsFont)
    button_HoughTestGUI_0.grid(column=0, row=0)

    button_HoughTestGUI_1 = tk.Button(root1, text='STEP 2A: Hough Detection Preview',
                                      command=HoughTestGUI_detectionPreview, font=buttonsFont)
    button_HoughTestGUI_1.grid(column=0, row=1)

    button_HoughTestGUI_2 = tk.Button(root1, text='STEP 3: Save Screenshot (3 uM file name)',
                                      command=HoughTestGUI_saveScreenshot_small, font=buttonsFont)
    button_HoughTestGUI_2.grid(column=1, row=1)

    button_HoughTestGUI_2 = tk.Button(root1, text='STEP 3: Save Screenshot (5uM file name)',
                                      command=HoughTestGUI_saveScreenshot_large, font=buttonsFont)
    button_HoughTestGUI_2.grid(column=1, row=2)

    text1 = Label(root1, text='Set parameters', font=buttonsFont)
    text1.grid(column=0, row=3)

    global canny_1
    canny_1 = tk.IntVar()
    canny_1 = tk.Scale(root1, from_=1, to=200, orient='horizontal')
    canny_1.set(16)
    canny_1_label = ttk.Label(root1, text='Canny Edge Detector (Param 1)')
    canny_1.grid(column=0, row=4)
    canny_1_label.grid(column=0, row=5)

    global center_1
    center_1 = tk.IntVar()
    center_1 = tk.Scale(root1, from_=1, to=250, orient='horizontal')
    center_1.set(14)
    center_1_label = ttk.Label(root1, text='Center Detection Threshold (Param 2)')
    center_1.grid(column=1, row=4)
    center_1_label.grid(column=1, row=5)

    global minimum_radius_1
    minimum_radius_1 = tk.IntVar()
    minimum_radius_1 = tk.Scale(root1, from_=0, to=50, orient='horizontal')
    minimum_radius_1.set(12)
    minimum_radius_1_label = ttk.Label(root1, text='Minimum Circle Radius (smaller)')
    minimum_radius_1.grid(column=0, row=6)
    minimum_radius_1_label.grid(column=0, row=7)

    global maximum_radius_1
    maximum_radius_1 = tk.IntVar()
    maximum_radius_1 = tk.Scale(root1, from_=0, to=100, orient='horizontal')
    maximum_radius_1.set(15)
    maximum_radius_1_label = ttk.Label(root1, text='Maximum Circle Radius (smaller)')
    maximum_radius_1.grid(column=1, row=6)
    maximum_radius_1_label.grid(column=1, row=7)

    global minimum_distance_1
    minimum_distance_1 = tk.IntVar()
    minimum_distance_1 = tk.Scale(root1, from_=1, to=100, orient='horizontal')
    minimum_distance_1.set(39)
    minimum_distance_1_label = ttk.Label(root1, text='Minimum Distance btw. Centers')
    minimum_distance_1.grid(column=0, row=8)
    minimum_distance_1_label.grid(column=0, row=9)

    global blur_level_1
    blur_level_1 = tk.IntVar()
    blur_level_1 = tk.Scale(root1, from_=1, to=65, orient='horizontal')
    blur_level_1_label = ttk.Label(root1, text='Blur')
    blur_level_1.grid(column=1, row=8)
    blur_level_1_label.grid(column=1, row=9)

    text3 = Label(root1, text='Number of circles detected will print to console.', font=buttonsFont)
    text3.grid(column=0, row=17)

    text4 = Label(root1, text='Screenshots are saved as original file name + "houghGUIanalysis" + # of beads', font=buttonsFont)
    text4.grid(column=0, row=18)

    root1.mainloop()


def HoughTestGUI_selectTesterImage():
    global tester_image_name
    tester_image_name = fd.askopenfilename(title='Open a .PNG or .JPEG file')
    original_clean_img = cv.imread(cv.samples.findFile(tester_image_name), cv.IMREAD_COLOR)


def HoughTestGUI_detectionPreview():
    global img
    img = cv.imread(cv.samples.findFile(tester_image_name), cv.IMREAD_COLOR)

    global circles1_count

    circles1_del = None

    gray_1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_1 = cv.medianBlur(gray_1, blur_level_1.get())
    rows = gray_1.shape[0]

    global hough_dictionary_1
    hough_dictionary_1 = {
        'image': gray_1,
        'method': cv.HOUGH_GRADIENT,
        'dp': 1,
        'param1': int(canny_1.get()),
        'param2': int(center_1.get()),
        'minRadius': int(minimum_radius_1.get()),
        'maxRadius': int(maximum_radius_1.get()),
        'minDist': int(minimum_distance_1.get()),
    }

    # Circle detection for "smaller" set
    circles1 = cv.HoughCircles(**hough_dictionary_1)

    if circles1 is not None:
        circles_found1a = len(circles1[0, :])
    else:
        circles_found1a = 0

    print("Circles found, before duplicate check:" + str(circles_found1a))

    circles1_delete_list = []

    if circles1 is not None:

        for idx_a, i in enumerate(circles1[0]):
            x_a = int(i[0])
            y_a = int(i[1])
            radius_a = int(i[2])

            for idx_b, j in enumerate(circles1[0][idx_a + 1:], start=idx_a +1):
                x_b = int(j[0])
                y_b = int(j[1])
                radius_b = int(j[2])

                inequality_left = (((x_b - x_a) ** 2) + ((y_b - y_a) ** 2))
                inequality_right = ((radius_a) ** 2)

                if inequality_left < inequality_right:
                    circles1_delete_list.append(idx_b)

        circles1_delete_list_refined = np.unique(circles1_delete_list)
        circles1_delete_list_refined = np.array(circles1_delete_list_refined, dtype=int)
        print("Circles to delete:" + str(len(circles1_delete_list_refined)))

        if circles1_delete_list_refined is not None:
            circles1_del = np.delete(circles1, circles1_delete_list_refined, axis=1)
        else:
            circles1_del = None

    print("NOTE: Duplicate detection and overlap removal does not account for actually overlapping beads which might be distinguishable by a human.")

    circles1_count = 0

    if circles1_del is not None:
        circles1 = np.uint16(np.around(circles1_del))
        for i in circles1[0]:
            circles1_count += 1
            center = (i[0], i[1])
            # Circle center
            cv.circle(img, center, 1, (0, 100, 100), 1)
            # Circle outline
            radius = i[2]
            cv.circle(img, center, radius, (0, 0, 255), 2)
        print("'Bead total after duplicate/overlap check: " + str(circles1_count) + " circles detected")
    else:
        print("'After duplicate/overlap check: No circles detected\n\n\n")

    cv.imshow("detected circles", img)

def HoughTestGUI_saveScreenshot_small():
    global img

    base_name = os.path.splitext(os.path.basename(tester_image_name))[0]
    default_name = f"{base_name}_houghGUIanalysis_"+str(circles1_count)+"x3um_beads.png"

    save_path = fd.asksaveasfilename(
        initialfile=default_name,
        defaultextension=".png",
        filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")],
    )

    if save_path:
        try:
            cv.imwrite(save_path, img)
            messagebox.showinfo("Success", f"Image saved successfully to:\n{os.path.basename(save_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving image:\n{str(e)}")

def HoughTestGUI_saveScreenshot_large():
    global img

    base_name = os.path.splitext(os.path.basename(tester_image_name))[0]
    default_name = f"{base_name}_houghGUIanalysis_"+str(circles1_count)+"x5um_beads.png"

    save_path = fd.asksaveasfilename(
        initialfile=default_name,
        defaultextension=".png",
        filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")],
    )

    if save_path:
        try:
            cv.imwrite(save_path, img)
            messagebox.showinfo("Success", f"Image saved successfully to:\n{os.path.basename(save_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving image:\n{str(e)}")

if __name__ == "__main__":
    main()