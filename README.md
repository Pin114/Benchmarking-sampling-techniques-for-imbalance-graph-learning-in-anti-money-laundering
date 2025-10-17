## Benchmarking sampling techniques for imbalance graph learning in anti-money laundering
### Research objectives:
The objective of this thesis is to study the current literature on sampling meth-
ods for graph representation learning and analyse their effect on models for
anti-money laundering.
### Description
Networks are a natural representation for many real-life problems, one of which
is transaction networks. Transactions and clients are often monitored using their
relationships with each other. On these transaction networks, many graph learn-
ing methods have been applied to try and fight money laundering [1]. Money
laundering is a specific form of financial fraud in which illegally obtained money
goes through multiple transactions and intermediaries to obscure the source of
the money and make it appear legitimate.
One of the problems hindering the development of potent detection models
is that fraud is uncommen and concealed, making the label distribution very
unbalanced [2]. Often, fewer than 1% of transactions are labelled as money
laundering. Previous research by Zhu et al. [3] has show that there are different
aspects influencing the power of sampling.
The aim of this thesis is to extend the work done by Deprez et al. [1], by pro-
viding a comprehensive review of sampling methods for network representation
learning and to study their effect on the performance on anti-money laundering
models.




## Folder structure

Use the folder structure provided with this repo to better structure your project and improve readability.  
There are few if not no reasons why you would want to deviate from this structure.  
Think about a few benefits:
- **Readability:** This structure is proven to be effective. Also, the more people use it, the easier it is to maneuver in other people's repos.  
- **Structure:** Your projects will be structured in a scalable form from day one. Even if it is a small side project, you never have to worry about reoganizing files.  
- **Efficiency:** Using this template will certainly save you some time now and in the future.

---

Here's the general structure with some additional info:

- `project/`  
*The main folder and root directory of your repo. This also is the directory to reference to within all your scripts and notebooks.*  
    - `assets/`  
    *Assets are all non-coding related files that you might want to save in your repo. These can include, e.g., pictures for your readme.md or PDFs*  
    - `config/`  
    *Any configuration file used for setting up experiments or training models.*  
    *It is good practice to save configurations or parameters for iteration in separate YAML files. We'll touch upon this in the sample scripts.*  
    *Getting into writing YAML scripts is super easy and will keep your code nice and clean (as we like it).*  
    - `data/`  
    *This is the only folder where you are going to store data (csv, pickle, or other).*  
    *You typically have two subfolders:* `raw/` *and* `processed/`.  
    *Put the preprocessing scripts in the* `scripts/` *folder (see below).*  
        - `processed/`  
        - `raw/`  
    - `lib/`  
    *Your library of other, of-the-shelf code.*  
    *When you do a benchmarking, e.g., and you have a method or package that you use without changing any of the code, put it here.*
    *If you do any tweaks to it, better to put it into * `src/`.
    *BTW notice that the* `lib/` *folder is typically included in the standard* `.gitignore` *file. Best to remove it and sync it with your project*
    - `notebooks/`  
    *Some love them, some hate them. Either way, do not just dump them in your root directory, but store them in their dedicated folder.*  
    *Depending on your project, you can create sub-folders, but try to stay as lean as possible.*  
    - `res/`  
    *The folder to store your results. Keep it neat and do not save them under your root directory.*
    - `scripts/`  
    *This is where you put all your runnables, i.e., if you do not use notebooks, but a single (or multiple) python scripts to run experiments, here is where they go.*  
    *As before: Subfolders are possible, but as few as possible.*
    *NOTE: Depending on how you run things, make sure that your root directory is properly defined 
    - `src/`  
    *The "source" directory. That's the flesh and bone of your project.*  
    *Here you will put all proprietary code that you use for your method.*  
    *Think about it this way: A script or notebook is just a way to tell your methods what to do. Its goal is to link hyperparameters, data, and methods in one spot.*  
    *The key to effective and good programming is to keep all these building blocks separatly for as long as possible.*  
    *In the end, your main scripts or notebooks will only be a few lines, saying: Use this dataset, with that method and these hyperparameters, and calculate this metric. End of script.*  
    *The* `src/` *directory has a few typical subfolders:*  
        - `data/`  
        *Scripts needed to load data from the* `data/` *directory.*  
        *There is no real data here, just code needed to make your data readable.*  
        *Why not load it in my main script or notebook? Because we want to be able to reuse it whenever we want and need.*  
        *That's why it is good practice to load data in form of a "data class". More on that in the sample scripts.*
            - `utils/`  
            *Any utility scripts.*
        - `methods/`  
        *Where all your proprietary methods will be stored.*  
            - `utils/`  
            *Any utility scripts for your methods.*
        - `utils/`  
        *Even more utility scripts? No - this folder contains anything from visualization scripts to tools for performance evaluation. It is anything that is not needed for training methods, or loading data.*
    - `LICENSE`  
    *When publishing code online, it is best to include a license. You can tweak the license in this project and add your name to it (it is the MIT license), or download a different template.*
    - `README.md`  
    *The README_PROJECT.md file is a template for your projects readme-file. After reading you can delete this readme and change the name of the template-readme to "README.md" so the right readme will display when pushing your code to GitHub*
    - `requirements.txt`  
    *Add the requirements (the packages and Python version necessary to run your code) to you repo.*  
    *This will help you and others whenever working with the repo in the future.*
