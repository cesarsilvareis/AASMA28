# AASMA - Emergency Response Problem
Autonomous Agents and Multi-Agent Systems 2022/23


## Authors:

* **César Reis** - *96849* - [cesarsilvareis](https://github.com/cesarsilvareis)
* **Henrique Vinagre** - *96869* - [Henrique Vinagre](https://github.com/henriquevinagre)
* **Yhya Dabah** - *96895* - [Henrique Vinagre](https://github.com/yhya96895)

___

## 📦 | Installation details:
The project was created in Python 3.10 and requires the installation of several libraries. To install the required dependencies, follow the steps below:
- We suggest you create a Python virtual environment using Python 3.10 or higher.
- Install with pip command all the library packages present in the ```requirements.txt``` file:
    ```
    pip install -r requirements.txt
    ```
___ 

## 🤖 | How to run our system:

Because our system was designed to make it easy to run tests, all you need to do is run the following command within our root project directory:

```
python3 run.py
```

If you want a custom execution, altering some default options, take a look at the options that we provide to you by running the previous command with the flag *help*:

```
python3 run.py --help
```

Keep in mind that the rendering option may take a while to finish and show the results of that execution.
For example, you may only want to see fast the results of each team without saving them:
```
python3 run.py --episodes 100 --no-rendering --show-results --no-save-results --max-generation-steps 50 --occupancy-map 4 --stressfulness 2
```

## ✍ | How to configure our system:

We also provide a unique configuration file - ```settings.py``` - which you can use to change some settings of our ERS environment, such as:

* The scenario style colors and the resource package path.
* Request priority probabilities and durations.
* Definition of occupancy map layouts for request generation. You can modify some of them according to our needs and finds.
