import scdiff.scdiff as S
def run_scdiff_init(data_file):
    E=S.TabFile(data_file).read('\t')
    #print E[0][:3]
    #global GL # Gene list global variable
    S.GL=E[0][3:]
    S.dTD={}
    S.dMb={}
    S.dTG={}
    E=E[1:]
    AllCells=[]

    for i in E:
            iid=i[0]
            ti=float(i[1])     # time point for cell i
            li=i[2]
            ei=[float(item) for item in i[3:]] # expression for cell i
            ci=S.Cell(iid,ti,ei,li) # cell i
            AllCells.append(ci)
    G1=S.Graph(AllCells,'auto',None)  #Cells: List of Cell instances 
    out_file = open('init_cluster_'+data_file+'.txt','w')
    pairs=[]
    pairs2=[]
    print 'G1'
    #print G1.clustering.dET
    #print G1.clustering.KET
    #cluster_cell_dict=defaultdict(lambda:[])
    for node in G1.Nodes:
        print 'ID: ', node.ID
        print 'index: ',G1.Nodes.index(node)
        ic=G1.Nodes.index(node)
        if node.P is not None:
            print 'P index: ',G1.Nodes.index(node.P)
            ip=G1.Nodes.index(node.P)
            pairs.append(str(ip)+' '+str(ic))
        else:
            print 'P index: ',None
        print 'ST: ',node.ST
        print 'T: ',node.T
        for cell in node.cells:
            print 'cell.ID: ', cell.ID
            #print 'cell.T: ', cell.T
            #print 'cell.Label: ', cell.Label
            #cluster_cell_dict[ic].append(cell.ID)
            pairs2.append(cell.ID+" "+str(ic))
    print pairs
    print pairs2
    out_file.write('\t'.join(pairs)+'\n')
    out_file.write('\t'.join(pairs2)+'\n')
#run_scdiff_init('treutlein2014')
#run_scdiff_init('treutlein2014_2copy')
