clear;clc
rootD = '/isilon/LFMI/VMdrive/Mugihiko/GlobalShape/Behav/DNNObjectRec_Kato2026';
addpath([rootD '/codes'])
savedir = [rootD '/analysis/Table1'];
if ~exist(savedir,'dir');mkdir(savedir);end
cnntype = {'resnet50','convnextL','cornet-s','vit_l_16',...
    'resnet50-sin','resnet50-blur-st','cornet-s-blur-st',...
     'clip_convnextL_image','clip_vit-l-laion_image',....
    'convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384' 'vit_large_patch14_clip_224.laion2b_ft_in12k_in1k'};

imtype    = {'Sparse' 'Dense'};
top   = 1;
for type = 1:length(imtype)
    out = cell(20,length(cnntype));
    for cnn = 1:length(cnntype)
        [dat,~,pred]= get_cnn_resp(rootD,cnntype{cnn},imtype{type},top);

        tbl = tabulate(pred);
        [~,ord] = sort(cell2mat(tbl(:,3)),'descend');
        tbl = tbl(ord,:);
        idx = cell2mat(tbl(:,3))>10;
        for ii = 1:sum(idx)
            out{ii,cnn} = [tbl{ii,1} ' (' sprintf('%0.1f',tbl{ii,3}) ')'];
        end
    end
    out = [cnntype; out];
    out(cellfun(@isempty,out)) = {' '}; 
    writecell(out,[savedir '/' imtype{type} '.csv'])

end
