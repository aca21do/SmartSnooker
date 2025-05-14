import cv2
import numpy as np
from scipy.spatial import KDTree
# TODO: create last pockets/cushions for stability
# use pockets to cut edges that are to long
# THEN pockets to extend edges that are to short



# ----------------------------------- functions -------------------------------

def preproccessing(frame):
    # ------------------------ define how to find the table mask --------------
    def find_coloured_area(frame_hsv, colour):
        binary_coloured_area = np.zeros((frame_hsv.shape[0],frame_hsv.shape[1]))
        
        # apply mask based on table colour
        if colour == "green":
            binary_coloured_area = cv2.inRange(frame_hsv, np.array([38,120,38]), np.array([150,255,255])) 
        elif colour == "blue":
            binary_coloured_area = cv2.inRange(frame_hsv, np.array([100,150,0]), np.array([140,255,255]))
        else:
            print("table not found")
            
        
        # open image (erode then dilate to remove small blobs)
        open_structure_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        binary_coloured_area = cv2.morphologyEx(binary_coloured_area, cv2.MORPH_OPEN, open_structure_element) # closes gaps 
        
        # close image (dilate and erode to close holes inside and gaps between objects)
        close_structure_element = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        binary_coloured_area = cv2.morphologyEx(binary_coloured_area, cv2.MORPH_CLOSE, close_structure_element)
        
        contours, hierarchy = cv2.findContours(binary_coloured_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.002*cv2.arcLength(largest_contour, True)# 0.002
        approx_contours = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        return approx_contours, binary_coloured_area
    # ----------------------------------------------------------------------
    
    # apply preproccessing
    frame_blurred = cv2.GaussianBlur(frame, (7,7), 1.5)
    frame_blurred_hsv = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV) # h: 0-179, s:0.255, v:0-255
    
    # find approximate rectangle of coloured table area
    coloured_area_contour, binary_coloured_area = find_coloured_area(frame_blurred_hsv, table_colour) #( (y,x) coords)
    
    # remove area outside of coloured area from frame
    coloured_area_contour_mask = np.zeros((frame_height,frame_width), np.uint8)
    cv2.drawContours(coloured_area_contour_mask, [coloured_area_contour], -1, (255),thickness=cv2.FILLED)
    coloured_area_contour_mask = coloured_area_contour_mask//255
    
    # these are the preproccessed frames to be passed into finctions
    frame_blurred = np.multiply(coloured_area_contour_mask[:,:,None], frame_blurred)
    frame_blurred_hsv = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV)
    frame_blurred_gray = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)
    
    return (frame_blurred, frame_blurred_hsv, frame_blurred_gray) ,coloured_area_contour, binary_coloured_area
    


def update_pocket_estimates(contours, frame_width, frame_height, current_estimates):# maybe change this to finding pockets via euclidian distance to corners
    contours = contours[:,0,:]
    
    # select the min and max coordinates from the contour surrounding the coloured area of the table
    min_x = np.min(contours[:,0])    
    min_y = np.min(contours[:,1])
    max_x = np.max(contours[:,0])
    max_y = np.max(contours[:,1])
    
    mid_x = max_x - min_x
    mid_y = max_y - min_y
    x_offset = mid_x//5
    y_offset = mid_y//5
    
    screen_corners = np.array([(0,0),(frame_width,0),(frame_width, frame_height),(0,frame_height)])
    
    # find the nearest point in the table contours to each outer corner
    contour_tree = KDTree(contours)
    distances, indicies = contour_tree.query(screen_corners)
    corner_neighbours = np.array(contours[indicies])
    
    list_of_top_left_coords  = np.array([(x,y) for (x,y) in contours if x < corner_neighbours[0][0] + x_offset and y < corner_neighbours[0][1] + y_offset])
    list_of_top_right_coords = np.array([(x,y) for (x,y) in contours if x > corner_neighbours[1][0] - x_offset and y < corner_neighbours[1][1] + y_offset])
    list_of_btm_right_coords = np.array([(x,y) for (x,y) in contours if x > corner_neighbours[2][0] - x_offset and y > corner_neighbours[2][1] - y_offset])
    list_of_btm_left_coords  = np.array([(x,y) for (x,y) in contours if x < corner_neighbours[3][0] + x_offset and y > corner_neighbours[3][1] - y_offset])
    
    
    # if pocket found, replace current pocket
    try:
        top_l_x = np.min(list_of_top_left_coords[:,0])
        top_l_y = np.min(list_of_top_left_coords[:,1])
    except:
        top_l_x = current_estimates[0,0]
        top_l_y = current_estimates[0,1]
        
    try:
        top_r_x = np.max(list_of_top_right_coords[:,0])
        top_r_y = np.min(list_of_top_right_coords[:,1])
    except:
        top_r_x = current_estimates[1,0]
        top_r_y = current_estimates[1,1]
    
    try:
        top_c_x = (top_l_x + top_r_x)//2
        top_c_y = (top_l_y + top_r_y)//2
    except:
        top_c_x = current_estimates[2,0]
        top_c_y = current_estimates[2,1]
        
    try:
        btm_r_x = np.max(list_of_btm_right_coords[:,0])
        btm_r_y = np.max(list_of_btm_right_coords[:,1])
    except:
        btm_r_x = current_estimates[3,0]
        btm_r_y = current_estimates[3,1]
    
    try:
        btm_l_x = np.min(list_of_btm_left_coords[:,0])
        btm_l_y = np.max(list_of_btm_left_coords[:,1])
        
    except:
        btm_l_x = current_estimates[4,0]
        btm_l_y = current_estimates[4,1]
    
    try:
        btm_c_x = (btm_l_x + btm_r_x)//2
        btm_c_y = (btm_l_y + btm_r_y)//2
        
    except:
        btm_c_x = current_estimates[5,0]
        btm_c_y = current_estimates[5,1]
    
    pockets = np.array([(top_l_x, top_l_y), (top_c_x,top_c_y),(top_r_x, top_r_y),
                     (btm_r_x, btm_r_y), (btm_c_x,btm_c_y),(btm_l_x, btm_l_y)])
    
    return pockets
    



def find_hough_lines_from_edges(edges):
    close_structure_element = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    edges2 = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_structure_element)
    
    lines=cv2.HoughLinesP(edges2, 1, np.pi/180, threshold=30, 
                          minLineLength=frame_height/12, maxLineGap=12)
    return lines[:,0]

    
def find_possible_cushions(lines, edges):
    
    angle_threshold = 10
    horizontal_lines = []
    vertical_lines = []

    if lines is not None:
        for line in lines:
            #print(line)
            x1, y1, x2, y2 = line
            #frame = cv2.line(frame, (x1,y1), (x2,y2), (0,255,255),1)
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 /np.pi)
            
            if angle<angle_threshold:
                horizontal_lines.append(line)
            elif angle> 90 - angle_threshold and angle < 90 + angle_threshold:
                vertical_lines.append(line)
                
                
    # label candidate cushions
    frame_height = edges.shape[0]
    frame_width = edges.shape[1]
    
    possible_cushions_array = []
    possible_top_cushions = [line for line in horizontal_lines if line[1] < frame_height//4 and line[3] < frame_height//4]
    possible_bottom_cushions = [line for line in horizontal_lines if line[1] > 3*frame_height//4 and line[3] > 3*frame_height//4]
    possible_left_cushions = [line for line in vertical_lines if line[0] < frame_width//5 and line[2] < frame_width//5]
    possible_right_cushions = [line for line in vertical_lines if line[0] > 4*frame_width//5 and line[2] > 4*frame_width//5]
    possible_cushions_array.append(possible_top_cushions)
    possible_cushions_array.append(possible_right_cushions)
    possible_cushions_array.append(possible_bottom_cushions)
    possible_cushions_array.append(possible_left_cushions)
    
    
    try:
        #possible_cushions = np.concatenate((possible_top_cushions, possible_bottom_cushions, possible_left_cushions, possible_right_cushions), axis=0)
        possible_cushions = [possible_top_cushions, possible_bottom_cushions, possible_left_cushions, possible_right_cushions]
    
    except:
        possible_cushions = np.array([])
        
    else:
        possible_cushions = possible_cushions_array
    return possible_cushions


def crop_cushion_lines(possible_cushion_lines, pockets):
    # this function searches for any cushion line end points
    # lying inside a pocket and moves them to the nearest point on the circumference

    for cushion_i, cushion in enumerate(possible_cushion_lines):
        for line_i, line in enumerate(cushion):
            x1, y1, x2, y2 = line
            
            
            try:
                for pocket_i, pocket in enumerate(pockets):
                    px, py, rad = pocket
                    
                    # 1st line end
                    x1_dist = px-x1
                    y1_dist = py-y1
                    center_to_point1 = np.sqrt((x1_dist*x1_dist) + (y1_dist*y1_dist))
                    
                    # if euclid distance < radius
                    if center_to_point1 < rad:
                        x1_normalized = x1_dist / center_to_point1
                        y1_normalized = y1_dist / center_to_point1
                        x1 = int(px - (x1_normalized * rad))
                        y1 = int(py - (y1_normalized * rad))
                         
                    # 2nd line end
                    x2_dist = px-x2
                    y2_dist = py-y2
                    center_to_point2 = np.sqrt((x2_dist*x2_dist) + (y2_dist*y2_dist))
                    
                    if center_to_point2 < rad:
                        x2_normalized = x2_dist / center_to_point2
                        y2_normalized = y2_dist / center_to_point2
                        x2 = int(px - (x2_normalized * rad))
                        y2 = int(py - (y2_normalized * rad))
                
                possible_cushion_lines[cushion_i][line_i] = [x1,y1,x2,y2]
                
            except:
                print("error cropping line")
                
    return possible_cushion_lines
 

def update_pockets(frame_hsv, edges,pocket_estimates, current_pockets):
    # improve by using the table colour threshold and finding the pockets as points where there isn't a straight line,
    # rather than searching for circles
    circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1.5,20,
                                param1=100,param2=30,minRadius=1,maxRadius=35)
     
    circles = np.uint16(np.around(circles))
    circles = circles[0,:,:]
    pocket_estimate_tree = KDTree(circles[:,:2])
    distances, indicies = pocket_estimate_tree.query(pocket_estimates)
    
    # update if pockets found
    try:
        pockets = np.array(circles[indicies])
    except:
        pockets = current_pockets

    return pockets


def update_balls(frame, frame_hsv, frame_gray, balls, cue_ball, pockets, binary_coloured_area):    
    """
    Bearing in mind there won't always be so many balls.
    Try removing the table color, by using a mask to search for the hole in the mask,
    rather than a greyscale image of the whole tble. This should fix the white circes of 
    striped balls being identified rather than the whole ball.
    """
    
    corner_pockets = np.array([pockets[0],pockets[2],pockets[3],pockets[5]])
    pockets_x_min = max(corner_pockets[0,0],corner_pockets[3,0])
    pockets_x_max = min(corner_pockets[1,0], corner_pockets[2,0])
    pockets_y_min = max(corner_pockets[0,1],corner_pockets[1,1])
    pockets_y_max = min(corner_pockets[2,1],corner_pockets[3,1])
    

    # find the cue ball, the find balls of a similar radius
    lower_white_threshold = np.array([0,0,150])
    upper_white_threshold = np.array([255,150,255])
    white_mask = cv2.inRange(frame_hsv, lower_white_threshold, upper_white_threshold)
    
    # open and close mask to remove noise and complete circles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations = 0)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations = 2)
    
    white_circles = cv2.HoughCircles(white_mask,cv2.HOUGH_GRADIENT,1.4,15,
                                param1=100,param2=15,minRadius=5,maxRadius=25)#[0]
    
    
    # Now some white circles have been returned, we need to find the cue ball,
    # as hough circles also returns the white areas on striped balls.
    
    cue_ball_candidates = []
    cue_ball_candidate_scores = []
    #cue_ball_candidate_contours = []
    
    # calculate a score for each ball based on roundness and coloured area
    if white_circles is not None:
        white_circles = white_circles[0]  
        
        for circle_i, circle in enumerate(white_circles):
            x,y,r = circle
            x=np.uint16(np.around(x))
            y=np.uint16(np.around(y))
            r=np.uint16(np.around(r))
            
            # find the detected circle's mask
            circle_mask = np.zeros_like(white_mask)
            cv2.circle(circle_mask, (x, y), r, 255, -1)
            
            # find the intensities of the area within the mask (1 is perfect)
            mean_intensity_after_morphology = cv2.mean(white_mask, mask=circle_mask)[0] /255
            mean_intensity_s = 1- (cv2.mean(frame_hsv, mask=circle_mask)[1]/255)
            mean_intensity_v = cv2.mean(frame_hsv, mask=circle_mask)[2]/255
            
            # find the countours to measure circulatiry
            contours, _ = cv2.findContours(circle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            max_circularity = 0
            #contour_i = None
            
            for i,contour in enumerate(contours):
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                circularity = (4 * np.pi * area) / (perimeter ** 2) # 1 for perfect circle
                if circularity > max_circularity:
                    max_circularity = circularity
                    #contour_i = i
                    
            #best_contour = contours[contour_i]
            
            cue_ball_candidate = [x,y,r,
                                  mean_intensity_after_morphology,
                                  mean_intensity_s,
                                  mean_intensity_v,
                                  max_circularity]

            candidate_score = ((mean_intensity_s + mean_intensity_v) *
                               max_circularity * r)
            
            #also output contours
            cue_ball_candidates.append(cue_ball_candidate)
            cue_ball_candidate_scores.append(candidate_score)
            #cue_ball_candidate_contours.append(best_contour)
    
        
        cue_ball_i = np.argmax(cue_ball_candidate_scores)
        cue_ball = np.uint16(np.around(white_circles[cue_ball_i]))
 
            
                
    # -----    find coloured balls     
    rad = cue_ball[2]
    rad_tolerance = 4
    
    balls = cv2.HoughCircles(frame_gray,cv2.HOUGH_GRADIENT,1.5,20, param1=50,param2=30,
                                minRadius= rad - rad_tolerance,
                                maxRadius= rad + rad_tolerance)
    
    if balls is not None:
        balls = np.uint16(np.around(balls))
        balls = balls[0,:,:]
    
        # if a matching ball is found in balls, update the cue ball to be this ball
        for i,ball in enumerate(balls):
            x,y,r = ball
            if (abs(x-cue_ball[0]) <rad and abs(y-cue_ball[1])<rad):
                #cue_ball = ball
                balls[i] = [0,0,0] # this is the cue ball
                
            if x < pockets_x_min or x > pockets_x_max or y < pockets_y_min or y > pockets_y_max:
                balls[i] = [-1,-1,-1]
            
    return balls, white_mask, cue_ball



def find_candidate_shots(play_area_mask, balls, cue_ball, pockets):
    # TODO:
    # Rename variables to match report
    # make shots check for cushion collisions so paralell shots close to the cushion
    # do not go through the cushion
    # 
    # implement `canon` shots
    cue_x, cue_y, cue_r = cue_ball
    
    
    # calculate single collision trajectories
    trajectories = [] # trajectories[target_ball, pocket] 
    wrong_trajectory_detected_pts = []
    
    if balls is not None:
        for target_ball in balls:
            t_x, t_y, t_r = target_ball
            t_x = int(t_x)
            t_y = int(t_y)
            t_r = int(t_r)
            
            if t_x < 1 or t_y < 1:# skips duplicate balls that have had their oordinates set to (-1,-1)
                continue
            
            for pocket in pockets:
                pocket_x, pocket_y, pocket_r = pocket
                pocket_x = int(pocket_x)
                pocket_y = int(pocket_y)
                pocket_r = int(pocket_r)
                
                # store the trajectory as lines (cue to target ball and target ball to pocket)
                target_to_pocket = [t_x, t_y, pocket_x, pocket_y]
                
                # find the trajectory from target ball to the pocket
                x_diff = pocket_x - t_x
                y_diff = pocket_y - t_y
    
                # normalise the trajectory to get a direction
                t_to_p_magnitude = np.sqrt((x_diff**2) + (y_diff**2))
                target_to_pocket_x_normalized = x_diff / t_to_p_magnitude
                target_to_pocket_y_normalized = y_diff / t_to_p_magnitude
                
                # find the center of the cue ball in the collision with target
                cue_at_collision_x = t_x - (target_to_pocket_x_normalized * (t_r + cue_r)) # wrong
                cue_at_collision_y = t_y - (target_to_pocket_y_normalized * (t_r + cue_r))
                
                # reject trajectory on opposite side of target ball
                # as this means the white ball has travelled through the ball
                cue_to_collision_x = cue_at_collision_x - cue_x
                cue_to_collision_y = cue_at_collision_y - cue_y
                
                cue_to_collision_magnitude = np.sqrt((cue_to_collision_x)**2 + (cue_to_collision_y)**2)
                cue_to_collision_x_normalized = cue_to_collision_x / cue_to_collision_magnitude
                cue_to_collision_y_normalized = cue_to_collision_y / cue_to_collision_magnitude
                
                x_on_trajectory = cue_at_collision_x - (cue_to_collision_x_normalized)
                y_on_trajectory = cue_at_collision_y - (cue_to_collision_y_normalized)
                
                dist_to_target_ball = np.sqrt((x_on_trajectory - t_x)**2 + (y_on_trajectory - t_y)**2)
                
                # remove `kissing` balls / balls inside of each other
                if dist_to_target_ball < t_r + cue_r:
                    continue
                
                
                # check if the shot is blocked by another ball (cue to target ball)
                try:
                    # interpolate along trajectory
                    trajectory_blocked = False
                    
                    # check if cue ball to target ball blocked
                    no_of_steps = int(cue_to_collision_magnitude/cue_r)
                    
        
                    if no_of_steps > 1:
                        interp_xs = np.arange(cue_x, cue_at_collision_x, (cue_at_collision_x - cue_x) / no_of_steps)
                        interp_ys = np.arange(cue_y, cue_at_collision_y, (cue_at_collision_y - cue_y) / no_of_steps)
                        
                        for interp_i in range(no_of_steps):
                            interp_x = interp_xs[interp_i]
                            interp_y = interp_ys[interp_i]
                            
                            for check_ball in balls:
                                if check_ball[0]==target_ball[0] and check_ball[1]==target_ball[1]:
                                    continue
                                
                                dist_to_check_ball = np.sqrt((interp_x - check_ball[0])**2 + (interp_y - check_ball[1])**2)
                                dist_to_cue_ball = np.sqrt((interp_x - cue_x)**2 + (interp_y - cue_y)**2)
                                if dist_to_check_ball < 1.8* cue_r and dist_to_cue_ball > cue_r*2.2:
                                    trajectory_blocked = True
                                    wrong_trajectory_detected_pts.append([interp_x,interp_y])
                                    
                    if trajectory_blocked:
                        continue
                              
                    # # check if the shot is blocked by another ball (target ball to pocket)
                    no_of_steps = int(t_to_p_magnitude/cue_r)
                    
                    if no_of_steps > 1:
                        interp_xs = np.arange(cue_at_collision_x, pocket_x, (pocket_x - cue_at_collision_x)/no_of_steps)
                        interp_ys = np.arange(cue_at_collision_y, pocket_y, (pocket_y - cue_at_collision_y)/no_of_steps)
                        
                        for interp_i in range(no_of_steps):
                            interp_x = interp_xs[interp_i]
                            interp_y = interp_ys[interp_i]
                            
                            for check_ball in balls:
                                if check_ball[0]==target_ball[0] and check_ball[1]==target_ball[1]:
                                    continue
                                
                                dist_to_check_ball = np.sqrt((interp_x - check_ball[0])**2 + (interp_y - check_ball[1])**2)
                                dist_to_cue_ball = np.sqrt((interp_x - cue_x)**2 + (interp_y - cue_y)**2)
                                if dist_to_check_ball < 1.8* cue_r and dist_to_cue_ball > cue_r*2.2:
                                    wrong_trajectory_detected_pts.append([interp_x,interp_y])
                                    trajectory_blocked = True
                                
                    if trajectory_blocked:
                        continue
                
                except:
                    print("error in checking clear trajectory path")
                    
                    
                # convert trajectory values to integers (pixels) to be displayed
                cue_at_collision_x = np.round(cue_at_collision_x).astype(np.int16)
                cue_at_collision_y = np.round(cue_at_collision_y).astype(np.int16)
                cue_to_collision = [cue_x,cue_y,cue_at_collision_x,cue_at_collision_y]
                
                # score the trajectory
                a = [target_to_pocket_x_normalized, target_to_pocket_y_normalized]
                b = [cue_to_collision_x_normalized, cue_to_collision_y_normalized]
                dot = np.dot(a,b)# / (a_mag * b_mag)
                
                cue_weight = 100*(1/200 + 1/cue_to_collision_magnitude)
                pocket_weight = 100*(2.5*(1/200 + 1/t_to_p_magnitude)* dot)
                score = cue_weight + pocket_weight
                
                trajectory = [cue_to_collision, target_to_pocket, score]
                trajectories.append(trajectory)
        
    trajectories = sorted(trajectories, key=lambda trajectories: trajectories[2], reverse=True)
    trajectories = trajectories
            
    return trajectories, wrong_trajectory_detected_pts

            
def find_play_area_mask(frame_hsv, pockets, possible_cushions):
    
    width = frame_hsv.shape[0]
    height = frame_hsv.shape[1]
    
    # label pockets
    tl = pockets[0]
    tc = pockets[1]
    tr = pockets[2]
    br = pockets[3]
    bc = pockets[4]
    bl = pockets[5]
    
    #extend play area into square pockets
    pockets_as_rectangles = []
    for pocket in pockets:
        x,y = pocket[0:2]
        rad = pocket[2]
        
        pocket_tl=[x-rad,y-rad]
        pocket_tr=[x+rad,y-rad]
        pocket_br=[x+rad,y+rad]
        pocket_bl=[x-rad,y+rad]
        
        pockets_as_rectangles.append(np.array([pocket_tl,pocket_tr,pocket_br,pocket_bl], dtype=np.int32))
    
    
    play_area_mask = np.zeros((width,height), dtype=np.uint16)
    play_area_rect = [
        [tl[0],tl[1]],
        [tr[0],tr[1]],
        [br[0],br[1]],
        [bl[0],bl[1]]]
    play_area_rect = np.array([play_area_rect], np.int32)
    play_area_mask = cv2.fillPoly(play_area_mask, play_area_rect, 255)
    
    for rect in pockets_as_rectangles:
        play_area_mask = cv2.fillPoly(play_area_mask, [rect], 255)
    
    return play_area_mask
    



def draw_overlays(frame, possible_cushions, coloured_area_contour, pocket_estimates, pockets, cushion_clusters, balls, cue_ball, play_area_mask, trajectories,wrong_pts):
    # select what to display in overlay for testing purposes
    display_play_area_mask = False
    display_cushion_lines = False
    display_pocket_estimates = False
    display_pockets = True
    display_balls = True
    display_trajectory = True
    display_all_trajectories = False
    display_blocked_shots = False
    
    
    
    if display_play_area_mask:
        mask_reshaped = np.zeros([frame.shape[0],frame.shape[1],frame.shape[2]], dtype=np.uint8)
        mask_reshaped[:,:,0] = play_area_mask
    
        #frame = cv2.addWeighted(frame, 0.7, mask_reshaped,0.3 , 0.0)
        frame_add = cv2.multiply(frame, mask_reshaped)
        frame = cv2.addWeighted(frame, 0.5, frame_add,0.5 , 0.0)
    
    
    if display_cushion_lines:
        for possible_cushion in possible_cushions:
            for line in possible_cushion:
                x1, y1, x2, y2 = line
                start = (x1,y1)
                end = (x2,y2)
                frame = cv2.line(frame, start, end, (250, 179, 102),1)
        
    
    if display_pocket_estimates:
        # draw pocket estiates (as box)
        first_x = pocket_estimates[0][0]
        first_y = pocket_estimates[0][1]
        for i in range(1,pocket_estimates.shape[0]):
            last_x = pocket_estimates[i-1][0]
            last_y = pocket_estimates[i-1][1]
            start = (last_x, last_y)
            
            next_x = pocket_estimates[i][0]
            next_y = pocket_estimates[i][1]
            end = (next_x, next_y)
            
            frame = cv2.line(frame, start, end, (255,255,255),1)
            if i+1 == pocket_estimates.shape[0]:#close box
                frame = cv2.line(frame, (first_x,first_y), end, (255,255,255),1)
        
        #draw pocket estimates (as points)
        for coords in pocket_estimates:
            x = coords[0]
            y = coords[1]
            frame = cv2.circle(frame, (x,y), radius=3, color=(0,128,255), thickness=-1)

        
    if display_pockets:
        for i in pockets:
            # draw the outer circle
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
        

    if display_balls:
        if balls is not None:
            for i in balls:
                # draw the outer circle
                cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
            
        # display cue ball
        cv2.circle(frame,(cue_ball[0],cue_ball[1]),cue_ball[2],(0,255,0),thickness=-1)


    if display_trajectory and trajectories != []:
        colours = [(0,255,255),(0,0,255),(255,255,0),(255,0,0),(0,255,0)]
        colour_i = 0
        
        
        for i,trajectory in enumerate(trajectories):
            
            if display_all_trajectories == True:
                prop_of_max_score = (len(trajectories) - i)/len(trajectories)
                colour = (255-int(prop_of_max_score*255),255,255-int(prop_of_max_score*255))
                thickness = 1+int(prop_of_max_score*5)
            else:
                colour = colours[colour_i % 5]
                thickness = 1
            
            line1, line2, _ = trajectory #[[cue, cue_at_collision],[target,pocket]]
            frame = cv2.line(frame, line1[0:2], line1[2:], colour, thickness)
            frame = cv2.line(frame, line1[2:], line2[2:], colour, thickness)
            frame = cv2.circle(frame, line1[2:], cue_ball[2],colour)
            
            colour_i +=1
            if display_all_trajectories == False:
                break
        
        
    if display_blocked_shots:
        for pts in wrong_pts:
            cv2.circle(frame, [int(pts[0]),int(pts[1])], 3,(255,0,0))

    return frame









# ----------------------------------- setup -----------------------------------
# TODO: finalise this
# params
table_colour = "green" # blue or green
camera_input = 1 # 0 = webcame, 1 = test video
test_video = 0
test_videos = [] # list of ("fileName.mp4", "colour")
test_videos.append(("test_0.mp4", "blue")) #0
test_videos.append(("test_1.mp4", "green")) #1
test_videos.append(("test_2.mp4", "green")) #2


# Replace "0" with a file path to work with a saved video
# setup video stream
if camera_input == 0:
    stream = cv2.VideoCapture(0) # webcam
if camera_input == 1:
    stream = cv2.VideoCapture("./assets/test_vids/" + test_videos[test_video][0]) # test video
    table_colour= test_videos[test_video][1]

if not stream.isOpened():
    print("Stream not found")
    exit()

fps = stream.get(cv2.CAP_PROP_FPS)
stream_width = int(stream.get(3))
stream_height = int(stream.get(4))

# set frame width and height to proccess at different resolutions for perforance
frame_width = stream_width//2
frame_height = stream_height//2
frame_no = 0

# setup video file saving
file_output = cv2.VideoWriter("assets/4_stream.avi",
            cv2.VideoWriter_fourcc(*'DIVX'),
            fps=fps, frameSize=(stream_width, stream_height))

# initialise arrays for features
cushion_lines = []
pocket_estimates = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
pockets = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
balls = []
cue_ball = [frame_width//2,frame_height//2, 15]



# ----------------------------------- MAIN LOOP -------------------------------
while True:
    # get camera input
    ret, original_frame = stream.read()

    
    if not ret: # if no frames are returned
        print("Stream ended")
        break
    
    resized_frame = cv2.resize(original_frame, (frame_width, frame_height)) # resized for performance ajustment
    
    # working_frame[0] = gaussian blur rgb colour space
    # working_frame[1] = gaussian blur hsv colour space
    # working_frame[2] = gaussian blur gray colour space
    working_frame, coloured_area_contour, binary_coloured_area = preproccessing(resized_frame)
    


    if frame_no % 15 == 0:
        # estimate pockets from coloured area
        # could this be improved by changing the parameters to find a shape most similar to a pool table?
        pocket_estimates = update_pocket_estimates(coloured_area_contour, frame_width, frame_height, pocket_estimates)
            
        # canny edge detection
        edges = cv2.Canny(working_frame[0], 1, 40)
        # find lines
        lines = find_hough_lines_from_edges(edges)
        
        # pockets
        pockets = update_pockets(working_frame[0], edges, pocket_estimates, pockets)
        # balls
        balls, edges, cue_ball = update_balls(working_frame[0], working_frame[1], working_frame[2], balls, cue_ball, pockets, binary_coloured_area)
    
        # catagorize lines into possible cusions
        possible_cushions = find_possible_cushions(lines,edges)# indexed clockwise
        possible_cushions = crop_cushion_lines(possible_cushions, pockets)
        
    
        # clusters (not using currently)
        cushion_clusters = []
        
        # play area
        play_area_mask = find_play_area_mask(working_frame[1], pockets, possible_cushions)
    
        # trajectories
        trajectories, wrong_pts = find_candidate_shots(play_area_mask, balls, cue_ball, pockets)
    
    # output frame
    annotated_frame = draw_overlays(resized_frame,
                                    possible_cushions,
                                    coloured_area_contour,
                                    pocket_estimates, pockets,
                                    cushion_clusters,
                                    balls,
                                    cue_ball,
                                    play_area_mask,
                                    trajectories,
                                    wrong_pts)
    
    output_frame = cv2.resize(annotated_frame, (stream_width, stream_height))
    file_output.write(output_frame)
    cv2.imshow("Webcam", output_frame)
    frame_no +=1
    
    # press q to exit program
    if cv2.waitKey(1) == ord('q'):
        break
    if cv2.waitKey(1) == ord('p'):
        
        while True:
            
            print("paused")
            if cv2.waitKey(1) == ord('p'):
                break

stream.release()
cv2.destroyAllWindows()