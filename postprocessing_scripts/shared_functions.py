import os
from os.path import join
import pandas as pd

#-------------------------------------------------------------------------------------------------------------------

def get_case_parameters(fluid, case, path_df_channel_edges, path_df_postproc, path_images, step_images, end_analysis):
    """
    Retrieves the postprocessing parameters for the specified case.

    Args:
    case (int): case number for which parameters are to be retrieved.
    step_images (int): step size for selecting images.
    end_analysis (str): method for determining the end of the analysis. 
                        For contact angle measurements, the analysis should stop when the cavities are reached.
                        For interface tracking, the analysis should stop when the interface leaves the ROI.

    Returns:
    tuple containing:
         - x_start (float): starting x position for the region of interest.
         - x_cavities (float): x position of the first cavity edge
         - y_channelEdge_bottom (int): y-coordinate of the bottom channel edge.
         - y_channelEdge_top (int): y-coordinate of the top channel edge.
         - framerate (int): frame rate for the case.
         - selected_images (list): list of selected image filenames based on the step size and end_analysis.
     """

    # get the channel edges
    df_channel_edges = pd.read_csv(path_df_channel_edges, index_col="case")
    y_channelEdge_top = int(df_channel_edges['channel_edge_top'][case])
    y_channelEdge_bottom = int(df_channel_edges['channel_edge_bottom'][case])

    # get the regoin of interest in x direction
    df_postproc =  pd.read_excel(path_df_postproc, sheet_name=fluid, index_col="case", skiprows=1) 
    x_start = df_postproc['xstart'][case]
    x_cavities = df_postproc['xcavity'][case]
    
    # get the framerate
    framerate = df_postproc['framerate'][case]
    
    # get start and stop frame 
    img_start = df_postproc['start'][case]
    if end_analysis == 'when_cavities_are_reached':
        img_end = df_postproc['cavitiesreached'][case]
    elif end_analysis == 'at_final_image':
        img_end = -1
    else:
        raise Exception("end_analysis is defined wrong.") 
    
    # create list of selected images
    images = [file for file in os.listdir(join(path_images, str(case))) if "jpg" or "png" in file][img_start:img_end]
    selected_images = images[::step_images] 
    
    return x_start, x_cavities, y_channelEdge_bottom, y_channelEdge_top, framerate, selected_images


#-------------------------------------------------------------------------------------------------------------------

def check_if_plot_now(plot_pictures, list_selected_images, current_index):
    """
    Checks if a plot should be generated based on the plot_pictures setting and the current index.

    Args:
    plot_pictures (str): setting for plotting pictures. 
                         It can be "all" to plot all pictures, or "last" to plot only the last picture,
                         otherwise no picture is plotted.
    list_selected_images (list): list of selected images.
    current_index (int): current index of the image being processed.

    Returns:
    bool: A boolean indicating whether a plot should be generated based on the plot_pictures setting and the current index.
    """

    plot_now = False
    if plot_pictures == "all":
        plot_now = True
    elif plot_pictures == "last" and current_index == len(list_selected_images)-1:
        plot_now = True
    return plot_now