# Merge-Sort-Multiple-Iterations


This is the parallel CUDA implementation of merge sort.

Please view the **"Merge Sort.pptx" file for detailed description and explanation** of project.


# CORE ALGORITHM:
    1)SPLIT INPUT ARRAY INTO SECTIONS OF SPECIFIC SIZE    
    2)PARALLELY SORT EACH INDIVIUAL SECTION USING A SINGLE BLOCK OF THREADS   
    3)ITERATIVELY MERGE SORTED SECTIONS IN ORDER TO PRODUCE BIGGER SECTIONS, UNTIL FINALLY ENTIRE ARRAY IS SORTED.    
        ALL MERGES IN A STAGE ARE PARALLELLY DONE BY MULTIPLE PARALLEL BLOCKS    
        
        
        
### References:   
    1)CUDA Zone | NVIDIA Developer (https://developer.nvidia.com/cuda-zone)   
    2)Udacity CS344: Intro to Parallel Programming (http://www.udacity.com/wiki/CS344)
    3)Bozidar, Darko & Dobravec, Tomaž. (2015). Comparison of parallel sorting algorithms. 
