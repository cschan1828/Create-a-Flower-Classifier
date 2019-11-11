import argparse
import utility_function
import model_function

def main():
    
    # Parse arguments
    in_args = utility_function.predit_get_input_args()
    
    # Load Model
    model = model_function.load_checkpoint(in_args.checkpoint)
    
    # Predicting Classes
    probs,classes = utility_function.predict(in_args.input, model, in_args.topk, in_args.gpu)
    
    # Map the Class Values
    ClassName = utility_function.NameConversion(classes,in_args.category_names)
    
    # Displaying class names
    utility_function.Display(ClassName,in_args.topk)

if __name__=="__main__":
    main()