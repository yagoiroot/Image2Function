import copy

import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import difflib
import gplearn
from gplearn.genetic import SymbolicRegressor
from joblib import dump, load
from sklearn.cluster import DBSCAN
import sympy as sp
from sympy.parsing.latex import parse_latex

scrip_dir = os.path.abspath(__file__)  # gets the BPL_code.py path
proj_dir = scrip_dir.replace(r'\Image2Function.py', '')


# requires antlr4-python3-runtime version 4.11.0
# WILL NOT WORK WITH ANY OTHER VERSION
# pip install antlr4-python3-runtime


def file_checker(file_name, file_type):
    """
    As opposed to file_checker(), this sunction searches through the full BPL Lab directory structure to try and find
    the specified file.
    Args:
        file_name: str, full file path of the desired file.
                    OR
                    name of the file without file extension, if the file is in the project directory.
        file_type: str, the type of file to be searched, can include the '.' so '.csv' or 'csv' are both valid.
                    OR
                    A list of file types to be searched.

    Returns:
        The full file path of the desired file, if it exists. If the file cannot be found then an error will be raised
        with suggestions if a similar file name is found.

    """
    scrip_dir = os.path.abspath(__file__)  # gets the BPL_code.py path
    proj_dir = scrip_dir.replace(r'\Event-Camera-Code\BPL_code.py','') #gets top level folder of the project
    #in my case 'BPL Lab'
    # print(f"proj_dir: {proj_dir}")

    # Check if the file_name is already the full file path
    if os.path.isfile(file_name):
        print(f"Looking at file '{file_name}'.")
        return file_name #already done
    # If file_name is not a complete file path already
    try:
        all_files = []
        for root, dirs, files in os.walk(proj_dir): #gets all files in the project directory
            for file in files:
                all_files.append(os.path.join(root, file))
        # print(f"all_files: {all_files}")
    except FileNotFoundError:
        raise FileNotFoundError(f"The directory {proj_dir} does not exist.") #Should be impossible to happen but idk
    # Filter only files of the same type
    if type(file_type)==str:
        same_type_files = [file for file in all_files if file_type in str(file)]
    if type(file_type)==list:
        same_type_files = [file for file in all_files if any(ext in str(file) for ext in file_type)]
    # print(f"same_type_files: {same_type_files}")
    file_found=False
    for file in same_type_files:
        # print(file)
        curr_file_1=os.path.basename(file)
        curr_file_name, curr_file_extension = os.path.splitext(curr_file_1)
        # print(f"curr_file_name: {curr_file_name}")
        # print(f"curr_file_extension: {curr_file_extension}")
        if file_name==curr_file_name and 'tmp' not in curr_file_extension:
            file_found=True
            print(f"Looking at file '{file}'.")
            return file

    if file_found==False: #if no exact matches have been found look for suggestions.
        # Find the closest match using difflib
        same_type_files_base=[os.path.basename(file) for file in same_type_files]
        # print(f"same_type_files_base: {same_type_files_base}")
        closest_matches = difflib.get_close_matches(file_name, same_type_files_base, n=1, cutoff=0.6)
        # Raise FileNotFoundError with suggestion if a close match is found
        if closest_matches:
            suggestion = closest_matches[0]
            suggestion_name, suggestion_extension = os.path.splitext(suggestion)
            raise FileNotFoundError(
                f"The file '{file_name}' with file type {file_type} does not exist. Did you mean '{suggestion_name}' with file type '{suggestion_extension}'?"
            )
        else:
            # If no close matches are found, raise error without suggestion
            raise FileNotFoundError(f"The file '{file_name}' does not exist and no similar files were found.")


def image_reader(input_image_location):
    """
    Reads an image from the specified location, processes the image to ensure it is 
    square, and extracts pixel coordinates based on intensity thresholds. This 
    function prepares the data for further image analysis or plotting.

    :param input_image_location: The file path to the input image.
    :type input_image_location: str
    :return: A list containing x-coordinates, y-coordinates, image width, image 
        height, and the prepared image data for plotting.
    :rtype: list
    :raises ValueError: If the image is not square.
    """
    exts = Image.registered_extensions()
    supported_extensions = [ex for ex, f in exts.items() if f in Image.OPEN]

    input_image_location = file_checker(input_image_location, supported_extensions)
    img = Image.open(input_image_location).convert('L')

    img_for_plot=plt.imread(input_image_location)
    # img.show()
    # print(img.size)
    # data = np.asarray(img.getdata()).reshape(img.size)
    data = np.array(img)
    print(data)

    image_dim=data.shape
    if image_dim[0] != image_dim[1]:
        raise ValueError("Image must be square.")
    print(f"image_dim[0]: {image_dim[0]}")
    print(f"image_dim[1]: {image_dim[1]}") ##1

    data[data > 120] = 255 #100 seems a functional cutoff for now.

    # x_vals=np.zeros(image_dim[0])
    # y_vals=np.zeros(image_dim[1])
    # x_vals=[0 for x in range(image_dim[0]) ]
    # y_vals=[0 for y in range(image_dim[1]) ]
    x_vals=[]
    y_vals=[]

    for x in range(image_dim[0]):
        for y in range(image_dim[1]):
            if data[y,x] < 255 :
                x_vals.append(x)
                # x_vals[x]=x
                y_vals.append(y)
                # y_vals[y]=y

    if len(x_vals)!= image_dim[0]:
        for i in range(image_dim[0]-len(x_vals)):
            x_vals.append(None)
    if len(y_vals)!= image_dim[1]:
        for i in range(image_dim[1]-len(y_vals)):
            y_vals.append(None)
    # print(f'x_vals: {x_vals}')
    # print(f'y_vals: {y_vals}')
    print(len(x_vals))
    print(len(y_vals))
    # y_vals=y_vals[::-1]
    x_vals=np.array(x_vals)
    y_vals=np.array(y_vals)
    mirror_axis = image_dim[1]/2
    y_vals=2*mirror_axis-y_vals
    # plt.imshow(data, cmap='grey')
    # plt.show()

    return [x_vals, y_vals, image_dim[0], image_dim[1], img_for_plot]

def image_fitter(input_image_location, x_domain=[0,1], y_range=[0,1], model_path=None,
                 use_saved=True, intd_expr=None):
    step1=image_reader(input_image_location)
    x_vals=step1[0]
    y_vals=step1[1]
    img_x_domain=step1[2]
    img_y_range=step1[3]
    img_for_plot=step1[4]

    x_vals=abs(x_domain[1]-x_domain[0])*x_vals*(1/img_x_domain)
    y_vals=abs(y_range[1]-y_range[0])*y_vals*(1/img_y_range)
    print(f"x_vals: {x_vals}")
    print(f"y_vals: {y_vals}")

    x_vals_w_outliers=copy.deepcopy(x_vals)
    y_vals_w_outliers=copy.deepcopy(y_vals)

    # use DBSCAN to remove outliers
    xy_vals=np.column_stack((x_vals,y_vals))
    eps=abs(x_domain[1]-x_domain[0])/100
    DB=DBSCAN(eps=eps, min_samples=10).fit(xy_vals)
    mask_inliers = (DB.labels_ != -1)
    x_vals, y_vals = x_vals[mask_inliers], y_vals[mask_inliers]

    #handle the case that the attempt to remove outlier with DBSCAN actually removed all the data
    if x_vals.size==0 or y_vals.size==0:
        print("OOPS! DBSCAN removed all the data. Using the original data instead.")
        x_vals=x_vals_w_outliers
        y_vals=y_vals_w_outliers


    preds=genetic_fitter(x_vals, y_vals , x_domain, y_range, model_path, use_saved)
    x_vals_pred=preds[0]
    y_vals_pred=preds[1]
    gp_prog=preds[2]

    latex_str=SymbRegg_to_latex(gp_prog)

    if type(intd_expr)==str:
        step2=Latex_to_function(intd_expr, x_domain, y_range)
        x_vals_intd=step2[0]
        y_vals_intd=step2[1]

    plt.rcParams.update({'font.size': 16})  # Set larger font size
    plt.rcParams["font.family"] = "Times New Roman"  # Use Times New Roman font
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.axisbelow'] = True  # Place axis grid lines behind plot elements

    # Create a new figure with specified size
    plt.figure(
        figsize=(6, 6),  # Width and height in inches
        tight_layout=True,  # Automatically adjust subplot parameters for better fit
        dpi=200,  # Optional higher resolution for saving (commented out)
    )

    plt.gca().set_aspect('equal')

    plt.imshow(img_for_plot, extent=[x_domain[0], x_domain[1], y_range[0], y_range[1]], cmap='gray', label='Image')
    # plt.plot(x_vals_w_outliers, y_vals_w_outliers, label='data w/outliers', marker='o', markersize=3, linestyle='')
    # plt.plot(x_vals, y_vals, label='data', marker='o', markersize=1, linestyle='')
    plt.plot(x_vals_pred, y_vals_pred, label=f'Prediction: ${latex_str}$', linestyle='--',
             color='tab:red', linewidth=2)
    if type(intd_expr)==str:
        plt.plot(x_vals_intd, y_vals_intd, label=f'Intended Function: ${intd_expr}$', linestyle='-',
                 color='deepskyblue', linewidth=2)

    plt.xlim(x_domain)
    plt.ylim(y_range)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(f'Drawn Plot & Fitted Function')
    plt.legend()
    plt.grid()
    plt.show()

def genetic_fitter(x_vals, y_vals, x_domain=[0,1], y_range=[0,1], model_path=None,
                   use_saved=True):
    """
    Performs symbolic regression using a genetic programming approach to fit a given set
    of x and y values. Optionally loads a saved model if available or trains a new model
    and saves it. The method also returns a prediction based on the evolved expression.

    :param x_vals: Array-like object containing input feature data to be fitted.
    :param y_vals: Array-like object containing target values corresponding to x_vals.
    :param x_domain: A list representing the domain range [min, max] for the input values. 
                      Default is [0, 1].
    :param y_range: A list representing the range of output values [min, max] the model might target. 
                     Default is [0, 1].
    :param model_path: Optional relative path to a pre-trained model file. If not provided, 
                       a default file location is used.
    :param use_saved: Boolean flag indicating whether to load a pre-trained model, if available. 
                       Default is True.
    :return: A list containing predicted x values, predicted y values, and the best evolved 
             expression (as a symbolic program object).
    """
    default_model_path=proj_dir+'\saved_model.joblib'
    if model_path==None:
        model_path=default_model_path
    else:
        model_path=proj_dir+model_path

    if use_saved and model_path and os.path.isfile(model_path):
        est_gp = load(model_path)
        print(f"Loaded pre-trained model from {model_path}")
    else:
        est_gp = SymbolicRegressor(
            population_size=1000,  # Number of candidate solutions in each generation
            generations=20,  # How many iterations (generations) to evolve
            tournament_size=20,  # Size of tournament for selecting individuals
            stopping_criteria=0.00001,  # Stop if error goes below this threshold
            const_range=(-1., 1.),  # Range for constant values in the expressions
            init_depth=(2, 6),  # Range for initial tree depths
            init_method='half and half',  # Method for generating the initial population
            function_set=['add', 'sub', 'mul', 'div', 'sqrt'],  # Operators to use in expressions
            metric='mean absolute error',  # Metric to optimize
            p_crossover=0.7,  # Crossover probability
            p_subtree_mutation=0.1,  # Subtree mutation probability
            p_hoist_mutation=0.05,  # Hoist mutation probability
            p_point_mutation=0.1,  # Point mutation probability
            verbose=1,  # Print progress during evolution
            parsimony_coefficient=0.02,  # Penalize overly complex expressions
            random_state=0  # For reproducibility
        )

        #reshape the data for the genetic programming fit
        x_vals=x_vals.reshape(-1,1)
        est_gp.fit(x_vals, y_vals)

        if model_path:
            dump(est_gp, model_path)
            print(f"Saved trained model to {model_path}")

    print("Best evolved expression:", est_gp._program)
    # print(type(est_gp._program))

    x_vals_pred=np.linspace(x_domain[0],x_domain[1],1000)
    y_vals_pred = est_gp.predict(x_vals_pred.reshape(-1,1))

    return [x_vals_pred,y_vals_pred, est_gp._program]

def SymbRegg_to_latex(gp_prog):
    """
    Converts a gplearn._program._Program object representation of a symbolic 
    regression expression into its corresponding LaTeX form.

    This function receives a `gplearn._program._Program` object, extracts its 
    string representation, maps gplearn functions and terminals to Sympy 
    equivalents, and finally converts the parsed expression into LaTeX format 
    using Sympy's LaTeX functionality.

    :param gp_prog: A gplearn._program._Program object representing the 
        symbolic regression program.
    :type gp_prog: gplearn._program._Program

    :return: A string containing the LaTeX representation of the symbolic 
        regression expression.
    :rtype: str

    :raises TypeError: If the input is not of type `gplearn._program._Program`.
    """
    if type(gp_prog) != gplearn._program._Program:
        raise TypeError("Input must be a gplearn._program._Program object.")
    expr_str = str(gp_prog)
    x= sp.symbols('x')

    # Map gplearn functions and terminals to Sympy
    locals_dict = {
        'add': sp.Add,
        'mul': sp.Mul,
        'sub': lambda x, y: x - y,
        'div': lambda x, y: x / y,
        'sqrt': sp.sqrt,
        'log': sp.log,
        'X0': x,
        # 'x1': X1, 'x2': X2
    }

    # 5. Parse and convert to LaTeX
    sympy_expr = sp.sympify(expr_str, locals=locals_dict)
    latex_str = sp.latex(sympy_expr)

    print("LaTeX form:", latex_str)
    return latex_str

def Latex_to_function(latex_string, x_domain, y_range):
    """
    Converts a LaTeX string representation of a mathematical expression into a numerical
    function, evaluates it over a specified range of x-values, and returns the evaluated
    x and y values.

    This function takes in a LaTeX string, parses it into a symbolic expression, converts
    it into a numerical function, and computes the y-values over a specified domain of x-values.
    It also allows specifying an optional y-range for bounding the y-values.

    :param latex_string: A string containing the LaTeX representation of the
        mathematical expression.
    :type latex_string: str
    :param x_domain: A tuple specifying the start and end values of the x-domain
        over which the function should be evaluated (inclusive).
    :type x_domain: Tuple[float, float]
    :param y_range: An optional tuple specifying the lower and upper bounds for the
        y-values. If None, no bounding is applied. Defaults to None.
    :type y_range: Optional[Tuple[float, float]]
    :return: A list containing two elements - the first being an array of
        x-values evaluated over the specified domain, and the second being an
        array of the corresponding y-values.
    :rtype: List[numpy.ndarray]
    """
    x=sp.Symbol('x')
    expr=parse_latex(latex_string)
    f=sp.lambdify(x, expr, modules=['numpy'])
    x_values=np.linspace(x_domain[0],x_domain[1],1000)
    y_values=f(x_values)

    return [x_values, y_values]

# image_location=r"function_x.png"
image_location=r"function_x^{.5}.png"
image_fitter(image_location, model_path='x^{.5}_moded.joblib', use_saved=False, x_domain=[0,10], y_range=[0,10], intd_expr='\sqrt{x}')
