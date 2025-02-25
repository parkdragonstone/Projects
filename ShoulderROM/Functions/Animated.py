# Autor: Rafael van Loon (R.M.vanLoon-1@student.tudelft.nl)
# course : Human movment kinematics
# Makes Animation plots 
# 2d and 3d animation plot

# Autor: Rafael van Loon (R.M.vanLoon-1@student.tudelft.nl)
# course : Human movment kinematics
# Makes 3d plots
def Animate_3d(markers,speed,angle,axis, name,sr,connections):
#def Animate_3d(markers,t,R,angles,trigger,multi,differ,block):
    """
    imports the nessasery packages for the mathematics and for the visualisation.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.patches as patches
    import mpl_toolkits.mplot3d.art3d as art3d
    from matplotlib.transforms import Affine2D
    from matplotlib.patches import Rectangle, PathPatch
    from matplotlib.text import TextPath


    """
    create a figure object and inistaites the correct type and size to it.
    """    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    ax.grid()
    #ax.computed_zorder = False
    
    """
    create the point objects in the pre complied functions
    """ 
    
    lines = []
    for conn in connections:
        marker1, marker2 = conn
        line, = ax.plot([],[],[], 'o-', lw=2)
        lines.append(line)
        
    ax.text2D(0.05, 0.95, f"{name}", transform=ax.transAxes)
    
    """
    determens the center of all points. This is used to create a correct window.
    """ 
    center = 0
    for i in markers: 
        center = center + markers[i][:,0]
    center = center / len(markers)

    """
    creates the correct window for time sample one based on the trigger. 
    Trigger = 1 is for moving persons(I.E. walking). Trigger = 0 is for stationary movements(think of arm movmetns when someone is standing still)
    """ 
    view_len = max(center)
    
    ax.view_init(vertical_axis=axis,elev=10, azim=angle)
    ax.set(xlim3d=(center[0]-0.80*view_len, center[0]+0.80*view_len), xlabel='X Position [m]')
    ax.set(ylim3d=(center[1]-0.80*view_len, center[1]+0.80*view_len), ylabel='Y Position [m]')
    ax.set(zlim3d=(center[2]-0.80*view_len, center[2]+0.80*view_len), zlabel='Z Position [m]')
    

    """
    Creats the global refernace frame in the figure
    """ 
    ax.quiver(0, 0, 0, abs(0.250) , 0, 0,color='r')
    ax.quiver(0, 0, 0, 0, abs(0.250) , 0,color='g')
    ax.quiver(0, 0, 0, 0, 0, abs(0.250) ,color='b')
    a = -1
    
    """
    Creats a timer object which displayes the elapesed time(based on 120fps data)
    """ 
    time_angle_4 = ax.text2D(0.05, 0.80, '', transform=ax.transAxes)
    
    
    speed = int(speed * 6)
    
    """
    Here the amount of frame shown in the plot are determind.
    The plot now deisplyes 15 frames per sencond instead of 120.
    The reason for this is to cut down the caluctions to make it run smooth anyway.
    """ 
    marker_label = list(markers.keys())
    marker_length = len(markers[marker_label[0]][0])
    nums = np.linspace(0, marker_length-1, num=int(marker_length/speed))
    frames = int((marker_length/speed))
    time = np.linspace(0, marker_length/sr, num=marker_length)
    """
    This is the main function which gets looped over by the precomplied animation function in line 328.
    """ 
    
        
    def animate(num, markers, lines):
        num_est = int(nums[num])
        num = num*speed
        """
        Sets the points in the correct place and delets the last idetial point to keep the used memmory down.
        Also assigns the correct name in the legenda.
        """  
        for idx, conn in enumerate(connections):
            marker1, marker2 = conn
            x = [markers[marker1][0, num], markers[marker2][0, num]]
            y = [markers[marker1][1, num], markers[marker2][1, num]]
            z = [markers[marker1][2, num], markers[marker2][2, num]]
            lines[idx].set_data(x, y)
            lines[idx].set_3d_properties(z)
        """
        updates the time in the figure
        """ 
        time_angle_4.set_text('time  = %.1f' % time[num])
                     
        """
        Deterens the center again for the correct window
        """     
        center = 0
        
        for i in markers: 
            center = center + markers[i][:,num]
        center = center / len(markers)
        
        #nan_bool = np.isnan(center[0])
        #count = 10
        #if nan_bool == True:
            #center = 0
            #for i in range(len(markers)): 
                #center = center + markers[i][:,1317]
            #center = center / len(markers)
            
            
        #print(center)
        """
        Updates the viewing window at each sampled time frame
        """
        
        ax.set(xlim3d=(center[0]-0.80*view_len, center[0]+0.80*view_len))
        ax.set(ylim3d=(center[1]-0.80*view_len, center[1]+0.80*view_len))
        ax.set(zlim3d=(center[2]-0.80*view_len, center[2]+0.80*view_len))
       
    """
    The main line of code whihc uses all functions and pre setup varaibles to create the animation data which is saved in the varaible ani which is then exported
    """ 
    ani = animation.FuncAnimation(
        fig, animate, frames, fargs=(markers, lines), interval=50,repeat=True)
    # plt.legend()
    return ani
 
    


