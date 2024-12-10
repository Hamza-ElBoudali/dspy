import dsp
import numpy as np
import os
import string
import random
import time
import ujson
from datasets.fingerprint import Hasher

from sklearn.cluster import KMeans
from .teleprompt import Teleprompter
from dsp.modules.finetuning import finetune_hf
from .utils import optimal_k_silhouette


class ExpertEnsemble(Teleprompter):
    def __init__(self, vectorizer: dsp.BaseSentenceVectorizer = None, clustering_algorithm='kmeans', max_k=10):
        self.vectorizer = vectorizer or dsp.FastEmbedVectorizer()
        self.clustering_algorithm = clustering_algorithm
        self.max_k = max_k
        self.training_data_directory = "training_data_directory"

        if not os.path.exists(self.training_data_directory):
            os.makedirs(self.training_data_directory)

    def finetune(
        self,
        samples,
        cluster_id,
        *,
        target="t5-small",
        bsize=12,
        accumsteps=1,
        lr=5e-5,
        epochs=1,
        bf16=False,
        int8=False,
        peft=False,
        path_prefix=None,
    ):
    
        # Prepare finetune <prompt, completion> pairs.
        finetune_data = [dict(prompt=sample.question, completion=sample.answer) for sample in samples]

        #
        # Dump as files.
        #

        data = finetune_data
        hashed_name = str(cluster_id) + "." + Hasher.hash(data)
        output_path = os.path.join(self.training_data_directory, f"{hashed_name}.jsonl")
        print(output_path)

        with open(output_path, "w") as f:
            for line in data:
                f.write(ujson.dumps(line) + "\n")

        finetune_path = output_path

        #
        # Train!
        #
        compiler_config = {
            "save": "".join(
                random.Random(time.time()).choices(string.ascii_uppercase + string.digits, k=13),
            ),  # https://stackoverflow.com/a/2257449/1493011
            "peft": peft,
            "fp16": False,
            "bf16": bf16,
            "int8": int8,
            "fid": False,
            "rationale": False,
            "batch_size": bsize,
            "epochs": epochs,
            "gradient_accumulation_steps": accumsteps,  # 2,
            "lr": lr,
        }

        compiler_config["save"] = (
            os.path.join(path_prefix, compiler_config["save"]) if path_prefix else compiler_config["save"]
        )

        training_data_path = finetune_path
        compiler_config_ = dict(compiler_config)
        compiler_config_["save"] = compiler_config["save"] + "." + str(cluster_id)
        best_ckpt_path = finetune_hf(training_data_path, target, compiler_config_)

        print(f"#> Best checkpoint path: {best_ckpt_path} for {cluster_id}")
        return dsp.HFModel(model=target, checkpoint=best_ckpt_path)

    def compile(self, student, *, trainset):
        self.student = student.reset_copy()

        embeddings = self.vectorizer([example.question + " " + example.answer for example in trainset])

        if self.clustering_algorithm == 'kmeans':
            k = optimal_k_silhouette(embeddings, self.max_k)
            print("Num clusters: ", k)
            self.k = k
            kmeans = KMeans(n_clusters=k, random_state=0)
            cluster_labels = kmeans.fit_predict(embeddings)
            cluster_assignments = {cluster_id: [trainset[i] for i in np.where(cluster_labels == cluster_id)[0]] for cluster_id in cluster_labels}

            fine_tuned_models = []
            for cluster_id in range(k):
                cluster = cluster_assignments[cluster_id]
                print(f"Finetuning model on cluster {cluster_id} of {k}...")
                model = self.finetune(cluster, cluster_id)
                print(f"Finetuned model on cluster {cluster_id}")
                fine_tuned_models.append(model)

            self.student.kmeans = kmeans
            self.student.models = fine_tuned_models
        
        elif self.clustering_algorithm == 'hdbscan':
            raise NotImplementedError("HDBSCAN clustering is not yet implemented.")

        self.student.cluster_assignments = cluster_assignments


        def forward(samples, **kwargs):
            embeddings = self.vectorizer([example.question for example in samples])
            clusters = self.student.kmeans.predict(embeddings)

            outputs = []
            for i, sample in enumerate(samples):
                output = self.student.models[clusters[i]](sample.question, **kwargs)[0]
                outputs.append(output)

            return outputs
        
        self.student.forward = forward
        
        return self.student
    
    