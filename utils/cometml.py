# import comet_ml at the top of your file
from comet_ml import Experiment
import config


def comet_init(prj_name):

    # Create an experiment with your api key
    return Experiment(
        api_key = config.COMET_API_KEY,
        project_name = prj_name,
        workspace = config.COMET_WORKSPACE,
        log_code = config.COMET_LOG_CODE,
        disabled = config.COMET_DISABLED
    )

def log_parameters(comet_exp, network, pretrained_model, fine_tune_dataset, source_dataset, target_dataset, batch_size, iterations, epochs):
    try: 
        comet_exp.log_parameter("Network", network) if network is not None else None
        comet_exp.log_parameter("Pretrained model", pretrained_model) if pretrained_model is not None else None
        comet_exp.log_parameter("Fine tune dataset", fine_tune_dataset) if fine_tune_dataset is not None else None
        comet_exp.log_parameter("Source dataset", source_dataset) if source_dataset is not None else None
        comet_exp.log_parameter("Target dataset", target_dataset) if target_dataset is not None else None
        comet_exp.log_parameter("Batch size", batch_size) if batch_size is not None else None
        comet_exp.log_parameter("Iterations", iterations) if iterations is not None else None
        comet_exp.log_parameter("Epochs", epochs) if epochs is not None else None
    except Exception as e:
            print("Error: ", e)  


def log_metrics(comet_exp, loss_name, accuracy_name, loss, accuracy):
    try: 
        comet_exp.log_metric(loss_name, loss)
        comet_exp.log_metric(accuracy_name, accuracy)
    except Exception as e:
            print("Error: ", e) 

def set_comet_exp_name(experiment, topk, src_comb, num_srcs, tar_name):
    exp_name = ""
    if topk is not None:
        exp_name = "top" + str(topk)
    exp_name = exp_name + ("_comb" if src_comb else "_each") + "Srcs" + str(num_srcs) + "_Tar" + tar_name

    experiment.set_name(exp_name)