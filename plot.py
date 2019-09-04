import matplotlib.pyplot as pyplot
import numpy as np

def plot_distribution(x_data,y_data,file_name=None,plot_type=None,multiple=False,legend=None,legend_title=None,plot_title=None):
        """
        """
        index = 0
        figure = pyplot.figure(1)
        if plot_type == "Torsions":
           if file_name == None:
             file_name = "torsion_distribution.png"
           pyplot.title("Torsion Distribution")
           pyplot.xlabel("Torsion Angle (Degrees)")
        if plot_type =="Angles":
           if file_name == None:
             file_name = "bond_angle_distribution.png"
           pyplot.title("Bond Angle Distribution")
           pyplot.xlabel("Bond Angle (Degrees)")
        
        pyplot.ylabel("Counts")
        if multiple:
         x_data = np.array([[float(n) for n in x] for x in x_data])
         y_data = np.array([[float(n) for n in y] for y in y_data])
         for x,y in zip(x_data,y_data):
          line, = pyplot.plot(x,y)
          if legend:
            line.set_label(str(legend[index]))
          index = index + 1
         if legend != None:
          if legend_title != None:
           pyplot.legend(title=legend_title)
          else:
           pyplot.legend()
        else:
          x_data = np.array([float(x) for x in x_data])
          y_data = np.array([float(y) for y in y_data])
          if legend != None:
            line, = pyplot.plot(x_data,y_data)
            line.set_label(str(legend[index]))
          else:
            pyplot.plot(x_data,y_data)
        if plot_title != None:
          pyplot.title(plot_title)
        pyplot.savefig(file_name)
        pyplot.close()

        return
