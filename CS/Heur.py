def heuristic_consensus(self):
    res=[0]*len(self.seqs)
    max_score=-1
    partial=[0, 0]
    
    for i in range(self.seq_size(0)-self.motif_size):
        for j in range(self.seq_size(1)-self.motif_size):
            partial[0]=i
            partial[1]=j
            sc=self.score(partial)
            
            if sc>max_score:
                max_score=sc
                res[0]=i
                res[1]=j
                
    for k in range(2, len(self.seqs)):
        partial=[0]*(k+1)
            
        for j in range(k):
            partial[j]=res[j]
        max_score=-1
            
        for i in range(self.seq_size(k)-self.motif_size):
            partial[k]=i
            sc=self.score(partial)
                
            if sc>max_score:
                max_score=sc
                res[k]=i
                    
    return res
