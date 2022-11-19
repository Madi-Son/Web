from process_video import ImageSeries, ProcessImage

sequence1 = ProcessImage('input_sequence')
sequence1.blur(kernel_size = 1) #use value between 3 and 9
sequence1.canny(threshold1 = 10, threshold2 = 50)
sequence1.crop_span() #crops the image to the vertical edges of the span
sequence1.hough_linesP()
sequence1.overlay_lines()
sequence1.draw_line_of_entry(215)
sequence1.write_series()

