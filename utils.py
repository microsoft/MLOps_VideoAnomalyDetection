import argparse


def get_workspace():
    import os
    import json
    from azureml.core.authentication import ServicePrincipalAuthentication
    from azureml.core import Workspace

    base_dir = "."

    config_json = os.path.join(base_dir, "config.json")
    with open(config_json, "r") as f:
        config = json.load(f)

    try:
        svc_pr = ServicePrincipalAuthentication(
            tenant_id=config["tenant_id"],
            service_principal_id=config["service_principal_id"],
            service_principal_password=config["service_principal_password"],
        )
    except KeyError:
        print("Getting Service Principal Authentication from Azure Devops.")
        svc_pr = None
        pass

    ws = Workspace.from_config(path=config_json, auth=svc_pr)

    return ws


def disable_pipeline(pipeline_name="", dry_run=True):
    from azureml.pipeline.core import PublishedPipeline
    from azureml.pipeline.core.schedule import Schedule

    if dry_run:
        print("Dry run: only printing what would be done")
    else:
        print("Disabling pipelines")

    ws = get_workspace()

    # Get all published pipeline objects in the workspace
    all_pub_pipelines = PublishedPipeline.list(ws)

    # We will iterate through the list of published pipelines and
    # use the last ID in the list for Schedule operations:
    print("Published pipelines found in the workspace:")
    for pub_pipeline in all_pub_pipelines:
        if (
            pub_pipeline.name.startswith("prednet")
            and pub_pipeline.name == pipeline_name
            or pipeline_name == ""
        ):
            print("Found pipeline:", pub_pipeline.name, pub_pipeline.id)
            pub_pipeline_id = pub_pipeline.id
            schedules = Schedule.list(ws, pipeline_id=pub_pipeline_id)

            # We will iterate through the list of schedules and
            # use the last ID in the list for further operations:
            print(
                "Found these schedules for the pipeline id {}:".format(
                    pub_pipeline_id
                )
            )
            for schedule in schedules:
                print(schedule.name, schedule.id)
                if not dry_run:
                    schedule_id = schedule.id
                    print(
                        "Schedule id to be used for schedule "
                        "operations: {}".format(
                            schedule_id
                        )
                    )
                    fetched_schedule = Schedule.get(ws, schedule_id)
                    print(
                        "Using schedule with id: {}".format(
                            fetched_schedule.id
                        )
                    )
                    fetched_schedule.disable(wait_for_provisioning=True)
                    fetched_schedule = Schedule.get(ws, schedule_id)
                    print(
                        "Disabled schedule {}. New status is: {}".format(
                            fetched_schedule.id, fetched_schedule.status
                        )
                    )

            if not dry_run:
                print("Disabling pipeline")
                pub_pipeline.disable()


def upload_data(folder='UCSDped1'):
    import os

    ws = get_workspace()

    ds = ws.get_default_datastore()

    src_dir = os.path.join("./data/UCSD_Anomaly_Dataset.v1p2", folder)
    target_path = os.path.join("prednet/data/raw_data", folder)
    ds.upload(
        src_dir=src_dir,
        target_path=target_path,
    )

    with open("placeholder.txt", "w") as f:
        f.write(
            "This is just a placeholder to ensure that this path exists in the blobstore.\n"
            "The scheduler of the master pipeline checks whether the time stamp of this file has changed.\n"
            "This was last updated when this folder was uploaded: %s\n" % folder
        )

    ds.upload_files(
        [os.path.join(os.getcwd(), "placeholder.txt")],
        target_path="prednet/data/raw_data/",
        overwrite=True
    )


def delete_data_from_blob(prefix):
    from azure.storage.blob.blockblobservice import BlockBlobService

    ws = get_workspace()

    def_blob_store = ws.get_default_datastore()

    print("Deleting blobs from folder:", prefix)
    blob_service = BlockBlobService(
        def_blob_store.account_name, def_blob_store.account_key
    )

    generator = blob_service.list_blobs(
        def_blob_store.container_name, prefix=prefix
    )
    for blob in generator:
        if blob.name.endswith("mp4"):
            print("Deleting: " + blob.name)
            blob_service.delete_blob(def_blob_store.container_name, blob.name)

    generator = blob_service.list_blobs(
        def_blob_store.container_name, prefix=prefix
    )
    for blob in generator:
        print("Deleting: " + blob.name)
        blob_service.delete_blob(def_blob_store.container_name, blob.name)


def cancel_all_runs(exp_name, run_id=None):
    from azureml.core import Experiment
    from azureml.core import get_run

    ws = get_workspace()

    exp = Experiment(ws, exp_name)

    if run_id:
        r = get_run(experiment=exp, run_id=run_id, rehydrate=True)

        # check the returned run type and status
        print(type(r), r.get_status())

        # you can cancel a run if it hasn't completed or failed
        if r.get_status() not in ["Complete", "Failed"]:
            r.cancel()
    else:
        # if you don't know the run id, you can list all
        # runs under an experiment
        for r in exp.get_runs():
            run = get_run(experiment=exp, run_id=r.id, rehydrate=True)
            for c in run.get_children():
                for gc in c.get_children():
                    if (
                        gc.get_status() == "Running"
                        or gc.get_status() == "Queued"
                    ):
                        print(gc.id, gc.get_status())
                        gc.cancel()
                if c.get_status() == "Running" or c.get_status() == "Queued":
                    print(c.id, c.get_status())
                    c.cancel()
            if r.get_status() == "Running" or r.get_status() == "Queued":
                print(r.id, r.get_status())
                r.cancel()


def delete_models(model_names=[]):
    from azureml.core.model import Model

    ws = get_workspace()

    model_list = Model.list(ws)

    for model in model_list:
        if model.name in model_names or len(model_names) == 0:
            model.delete()


def main():
    parser = argparse.ArgumentParser(description="Process input arguments")
    parser.add_argument(
        "--disable-pipelines",
        action="store_true",
        dest="disable_pipelines",
        help="disable published pipelines",
    )

    args = parser.parse_args()

    if args.disable_pipelines:
        disable_pipeline(dry_run=False)
    else:
        parser.print_usage()


if __name__ == "__main__":
    # execute only if run as a script
    main()
