from process_video import ImageSeries, ProcessImage
from data import read_and_plot
#print the current time and see what is taking up all the efficiency

# sequence1 = ProcessImage('input_sequence') 
# sequence1.blur(kernel_size = 1) #use value between 3 and 9
# sequence1.canny(threshold1 = 10, threshold2 = 50)
# sequence1.crop_span() #crops the image to the vertical edges of the span
# sequence1.hough_linesP()
# sequence1.overlay_lines()
# sequence1.draw_line_of_entry(215)
# sequence1.write_series()

read_and_plot()

#cut input sequence
# $ ffmpeg -i input_video/test_input1.MOV -r 1/1 input_sequence/%04d.png
#cut output video
# $ ffmpeg -framerate 5 -i output_sequence/output_%d.png -c:v libx264 -r 30 output55.mp4