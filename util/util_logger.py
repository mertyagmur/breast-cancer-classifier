from datetime import datetime
import os

def val_log_saver(test_name, model_results, train_test_opt):
    now = datetime.now()
    date_time = now.strftime("%d_%m_%Y__%H_%M")

    if not os.path.exists("./test_results/"):
        os.makedirs("./test_results/")

    f = open("./test_results/" 
             + date_time 
             + "__" 
             + train_test_opt
             + ".txt", "w")

    f.write(test_name + "\n")
    f.write("------------------------------------------" + "\n")
    for val in model_results[train_test_opt]:
        # print(val)
        f.write(str(val)+"\n")

    f.close()