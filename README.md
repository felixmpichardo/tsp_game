# tsp_game
Project Exploring the Traveling Salesman Problem for Introduction to AI at the UMN, Spring 2020

## Traveling Salesman Problem (TSP)
Given a set of cities on a map, what is the fastest way to travel to all of them once and return back to your original location? This is a problem faced by many businesses today and in the past [1, 3], and although many people and businesses work with some route that satisfies the answer, this is known to be a hard problem to answer. It’s called the traveling salesman problem, one of the oldest problems in computer science, and it’s at the core of the P vs NP problem because no polynomial time algorithm is known to exist to solve it. However, there are many approximate solution algorithms [1, 3], and it is an important test case for many now popular algorithms [3].

## TSP Types
There are actually three main classes of the TSP problem:
- Euclidean TSP
- Metric TSP
- Graph TSP

The Euclidean TSP is essentially the problem mentioned above: points on a plane. The other variants (including ones beyond this classification) relax those restrictions. For example, the graph formulation of the Euclidean TSP is always complete because every you can get to any city from any other city. However, the graph TSP can have less edges.

There are reasons to believe that the Euclidean TSP is different than the other variants and may be easier to solve [1]. For example, the Euclidean TSP has various approximation algorithms, but the graph variant does not [2].

## References
[1] Cook, W. J. In pursuit of the traveling salesman: mathematics at the limits of computation. Princeton University Press, 2011.

[2] Dasgupta, S., Papadimitriou, C. H., and Vazirani, U. V. Algorithms. McGraw-Hill Higher Education, 2008.

[3] Simha, R. The traveling salesman problem (tsp), 2018. Last accessed 25 March 2020. URL: https://www2.seas.gwu.edu/~simhaweb/champalg/tsp/tsp.html
