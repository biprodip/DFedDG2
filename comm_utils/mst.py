import numpy as np

def get_mst(adj,source = None):
  INF = 9999999
  n_V = adj.shape[0]
  
  selected = [0]*n_V #np.zeros(n_V)
  #keep the selected connections(MST) in an adj mat
  mst_mat = np.zeros((n_V,n_V))
  print(f"Selected: {selected}")

  if source is not None:
    selected[source] = True
  else:
    selected[0] = True

  print(f"Selected: {selected}")

  # set number of edge to 0
  no_edge = 0
  # the number of egde in minimum spanning tree will be
  # always less than(V - 1), where V is number of vertices in
  print("Edge : Weight\n")
  
  while (no_edge < n_V - 1):
    minimum = INF
    x = -1
    y = -1
    for i in range(n_V):
      if selected[i]:
        for j in range(n_V):
          if ((not selected[j]) and adj[i][j]):  
            # not in selected and there is an edge
            if minimum > adj[i][j]:
              minimum = adj[i][j]
              x = i
              y = j
    mst_mat[x][y]=1
    mst_mat[y][x]=1
    
    print(str(x) + "-" + str(y) + ":" + str(adj[x][y]))
    selected[y] = True
    no_edge += 1
  return mst_mat