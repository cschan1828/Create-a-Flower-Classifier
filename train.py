import argparse
import utility_function
import model_function

def main():
    # Load Data
    dataloaders,datasets = utility_function.LoadData()
    
    # Parse arguments
    in_args = utility_function.train_get_input_args()
    
    # Set up Model
    model, criterion, optimizer = model_function.SetModel(in_args.arch,datasets['train'],in_args.hidden_units,in_args.learning_rate)
    
    # Train the Model
    model_function.do_deep_learning(model, dataloaders, in_args.epochs, 10, criterion, optimizer, in_args.gpu)
    
    # Save the Model
    model_function.SaveCheckPoint(model,in_args.arch,datasets['train'], optimizer, in_args.learning_rate, in_args.hidden_units, in_args.save_dir)
    
    # Test the Model
    model_function.check_accuracy_on_test(model,dataloaders['test'], in_args.gpu)
    
if __name__ == "__main__":
    main()