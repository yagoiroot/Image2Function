import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import difflib
from gplearn.genetic import SymbolicRegressor
from joblib import dump, load

scrip_dir = os.path.abspath(__file__)  # gets the BPL_code.py path
proj_dir = scrip_dir.replace(r'\Image2Function.py', '')

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
    exts = Image.registered_extensions()
    supported_extensions = [ex for ex, f in exts.items() if f in Image.OPEN]

    input_image_location = file_checker(input_image_location, supported_extensions)
    img = Image.open(input_image_location).convert('L')
    # img.show()
    # print(img.size)
    # data = np.asarray(img.getdata()).reshape(img.size)
    data = np.array(img)

    image_dim=data.shape
    if image_dim[0] != image_dim[1]:
        raise ValueError("Image must be square.")
    print(f"image_dim[0]: {image_dim[0]}")
    print(f"image_dim[1]: {image_dim[1]}")

    data[data > 110] = 255 #100 seems a functional cutoff for now.

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
    print(f'x_vals: {x_vals}')
    print(f'y_vals: {y_vals}')
    print(len(x_vals))
    print(len(y_vals))
    # y_vals=y_vals[::-1]
    x_vals=np.array(x_vals)
    y_vals=np.array(y_vals)
    mirror_axis = 671
    y_vals=2*mirror_axis-y_vals
    # plt.imshow(data, cmap='grey')
    # plt.show()

    return [x_vals, y_vals, image_dim[0], image_dim[1]]

def image_fitter(input_image_location, x_domain=[0,1], y_range=[0,1], model_path=None, use_saved=True):
    step1=image_reader(input_image_location)
    x_vals=step1[0]
    y_vals=step1[1]
    img_x_domain=step1[2]
    img_y_range=step1[3]

    x_vals=x_vals*(1/img_x_domain)
    y_vals=y_vals*(1/img_y_range)

    preds=genetic_fitter(x_vals, y_vals , x_domain, y_range, model_path, use_saved)
    x_vals_pred=preds[0]
    y_vals_pred=preds[1]


    plt.rcParams.update({'font.size': 16})  # Set larger font size
    plt.rcParams["font.family"] = "Times New Roman"  # Use Times New Roman font
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.axisbelow'] = True  # Place axis grid lines behind plot elements

    # Create a new figure with specified size
    plt.figure(
        # figsize=(6, 7.2),  # Width and height in inches
        tight_layout=True,  # Automatically adjust subplot parameters for better fit
        dpi=200,  # Optional higher resolution for saving (commented out)
    )

    plt.gca().set_aspect('equal')

    plt.plot(x_vals, y_vals, label='data')
    plt.plot(x_vals_pred, y_vals_pred, label='prediction')

    plt.xlim(x_domain)
    plt.ylim(y_range)
    plt.legend()
    plt.grid()
    plt.show()

def genetic_fitter(x_vals, y_vals, x_domain=[0,1], y_range=[0,1], model_path=None, use_saved=True):
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
            function_set=['add', 'sub', 'mul', 'div'],  # Operators to use in expressions
            metric='mean absolute error',  # Metric to optimize
            p_crossover=0.7,  # Crossover probability
            p_subtree_mutation=0.1,  # Subtree mutation probability
            p_hoist_mutation=0.05,  # Hoist mutation probability
            p_point_mutation=0.1,  # Point mutation probability
            verbose=1,  # Print progress during evolution
            parsimony_coefficient=0.01,  # Penalize overly complex expressions
            random_state=0  # For reproducibility
        )

        #reshape the data for the genetic programming fit
        x_vals=x_vals.reshape(-1,1)
        est_gp.fit(x_vals, y_vals)
        print("Best evolved expression:", est_gp._program)

        if model_path:
            dump(est_gp, model_path)
            print(f"Saved trained model to {model_path}")

    x_vals_pred=np.linspace(x_domain[0],x_domain[1],1000)

    y_vals_pred = est_gp.predict(x_vals_pred.reshape(-1,1))

    return [x_vals_pred,y_vals_pred]

image_location=r"function_x.png"
image_fitter(image_location, model_path='test1.joblib', use_saved=False)
