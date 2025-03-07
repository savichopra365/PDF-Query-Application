

# Conditional Generative Adversarial Networks (CGANs)
We have used CGANs as Conditional Generative Adversarial Networks (CGANs) are a powerful class of neural networks used for generating synthetic data that closely resembles real data, conditioned on additional information such as class labels. CGANs consist of two neural networks: a Generator that creates synthetic data from random noise and class labels, and a Discriminator that evaluates the authenticity of the data, distinguishing between real and synthetic samples while also considering the class labels. These networks are trained simultaneously in a competitive setting, where the Generator aims to fool the Discriminator, and the Discriminator aims to correctly identify real versus fake data. We fine-tune the classified data to map to STM32 required format (2G/4G/6G).

## Architecture of CGAN
![image](https://github.com/user-attachments/assets/ec142ce8-5795-4dbf-9290-8f0a3cf8536a)

### 1. Generator
The generator in Conditional Generative Adversarial Networks (CGANs) is responsible for creating fake data from random noise, conditioned on additional information such as class labels. It takes a latent noise vector and class labels (e.g., stationary, walk, run) as input and produces fake data that resembles the real data distribution.

### 2. Discriminator
The discriminator in Conditional Generative Adversarial Networks (CGANs) is a binary classifier that distinguishes between real and fake data, conditioned on the same additional information as the generator.
It takes fake data from the generator and real data from the dataset as input and outputs the probability that the input data is real (True/False) .

### 3. ST Classifier
The ST classifier in Conditional Generative Adversarial Networks (CGANs) is a pre-trained classifier on a specific dataset (ST Dataset) used to further validate the generated data. 
It takes fake data from the generator as input and outputs a classification result (Y/N)..

## Training Process

### Step 1: Generator to Discriminator
1. **Latent Noise:** The generator receives a latent noise vector as input.
2. **Synthetic Output:** The generator produces synthetic data.
3. **Discriminator:** The fake data is fed into the discriminator, which has been trained on a specific dataset (PAMAP2).
4. **Feedback Training:** The discriminator provides feedback to the generator to improve the quality of the generated data.

### Step 2: Generator to ST Classifier
1. **Latent Noise + Categories:** The generator receives a latent noise vector and class labels (e.g., stationary, walk, run).
2. **Synthetic Output:** The generator produces fake data conditioned on the class labels.
3. **ST Classifier:** The fake data is fed into the ST classifier, which has been pre-trained on the ST dataset.
4. **Backtracking:** The ST classifier provides feedback to the generator to ensure the generated data is realistic and matches the class labels.

**Loss function used** : **WGAN-GP loss**:  The WGAN-GP (Wasserstein GAN with Gradient Penalty) loss improves the stability of GAN training by penalizing the gradient norm of the discriminator's output with respect to its input, 
ensuring it stays close to 1. This helps enforce the Lipschitz constraint, leading to better convergence and more realistic generated data.

### Setup 
system requirements --PYTHON
key components to generate synthetic dataset -- our components
PAMAPP DATA generator , PAMAPP data differentiator ,Finetune generator
PAMAPP DATASET GENERATOR SPECIFIED TO ST ACCELEROMETER SPECIFICTIES (

**Setup for Python**
1. Create a virtual environment
```
python3 -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```
2. Install the required dependencies
```
pip install numpy tensorflow scipy matplotlib seaborn scikit-learn pandas
```
**Finetuned model for LSM6DSL​ Data Fabrication**
```
pamap_latest_gan/generator_model_finetuned_LSM.h5
```
**Generator Model**
```
pamap_latest_gan/generator_model.h5 
```

**Discriminator Model**
```
pamap_latest_gan/discriminator_model.h5 
```

**Finetuned Model**
```
pamap_latest_gan/generator_model_finetuned_LSM.h5 
```
**Command to run in command line**
```
python3 generateLsmSynthenticData.py --config configuration.json --out dataset
```




### References
https://arxiv.org/search/stat?searchtype=author&query=Goodfellow,+I+J
https://keras.io/
https://www.tensorflow.org/tutorials/generative/dcgan
https://arxiv.org/search/stat?searchtype=author&query=Xu,+B



