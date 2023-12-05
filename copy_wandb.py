import wandb
import json
import os

# Your wandb details
your_entity = "hyesul"
your_project = "idl-project-person"

# Public wandb details
public_entity = "rohanprasad"
public_project = "idl-project-person-facet"


# Initialize the wandb API
api = wandb.Api()


def serialize(value):
    """Serialize the value to a JSON-compatible format"""
    try:
        # Try to serialize the value to JSON
        json.dumps(value)
        return value
    except TypeError:
        # If the value is not serializable, convert it to string
        return str(value)


# Access the public project
public_project_runs = api.runs(f"{public_entity}/{public_project}")

for run in public_project_runs:
    # Create a new run in your project
    with wandb.init(project=your_project, entity=your_entity) as new_run:
        local_directory = f"/home/nas2_userH/hyesulim/Dev/2023/11785-f23-prj/faceted_visualization/outputs/{run.id}"
        os.makedirs(local_directory, exist_ok=True)

        # Get the run summary
        run_summary_dict = dict(run.summary)
        # Copy data from the public run to your run
        for key, value in run.summary.items():
            new_run.summary[key] = serialize(value)
        # Optionally, copy other data such as configs, tags, etc.
        new_run.config.update(run.config)
        new_run.tags = run.tags

        # copy image files under media folder
        all_files = run.files()
        media_files = [f for f in all_files if f.name.startswith("media/")]
        # media_files = run.files(prefix="media/")

        if media_files:
            for file in media_files:
                file_path = os.path.join(local_directory, file.name)
                # Download the file
                file.download(root=local_directory)
                # If you want to upload these images to your own project, you can do so here
                # For example, using wandb.log() to log these images as artifacts in your run
                wandb.log({"media": wandb.Image(file_path)})
        else:
            print(f"No media files found for run {run.id}")

        # Save the new run
        new_run.save()
