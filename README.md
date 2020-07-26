# #BuildwithAI Predictive Algorithm Challenge

<p style="text-align: center;">
  <img src="https://hackmakers-35566.web.app/images/Logo_black.png"
       alt="Hackathon Logo" />
</p>

<br>
<a href="https://www.youtube.com/watch?v=ujBxDDCPDtE">Video Presentation</a>
<br>

## Problem

The sudden appearance of COVID-19 has changed the world as we know it. Accurate predictions are a key tool to take proper decisions and foresee future issues.

## Our Solution

Our proposed predictive algorithm combines epidemiology models and genetic algorithms. As epidemiology model, we decided to use SEIRS: Acronym of susceptible , exposed, infectious, recovered and re-susceptible.

<p style="text-align: center;">
  <img href="https://raw.githubusercontent.com/ryansmcgee/seirsplus/master/images/SEIRS_diagram.png"
       alt="SEIRS DIAGRAM"/>
</p>

However, this model depends on a set of parameters and its resulting prediction changes based on them. So, we want to find the set of parameters that better fits the current curve of positive cases, expecting that the model will generalize to the following days we want to predict.

<p style="text-align: center;">
  <img href="parameters.gif"
       alt="Evolution of the parameters" /> 
</p>
![Parameters]()

We propose to estimate the best set of parameters based on a Genetic Algorithm. The Genetic Algorithm considers several sets of parameters, re-combines them, and makes mutations to look for the parameters that better fit the current curve.

We apply our proposed algorithm to each state separately, obtaining specialized models capable of handling the specific conditions of each state.

## Resutls

Finally, aggregating results from each state, we obtain our final prediction curve.

![Results](results.png)


## Team Make Unicorns Great Again

<p style="text-align:center; ">
<img src="teamlogo.png"
     alt="Logo of the team"
     style="weight: 50%; height: auto;" />
</p>

* Person 1
* Person 2
* Person 3
* Person 4
* Person 5
* Person 6
* Person t

<style>

  a:hover, a:visited{
    text-decoration: none;
  }
</style>
