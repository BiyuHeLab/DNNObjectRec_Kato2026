clear;clc
rootD = '/isilon/LFMI/VMdrive/Mugihiko/GlobalShape/Behav/DNNObjectRec_Kato2026';
addpath([rootD '/codes'])
savedir = [rootD '/analysis/Fig4_error_consistency'];
if ~exist(savedir,'dir');mkdir(savedir);end
load([rootD '/rawdata/behav_exp2/summary.mat'])
cnntype   = {'resnet50','convnextL','cornet-s','vit_l_16',...
    '','resnet50-blur-st','cornet-s-blur-st','resnet50-sin',...
    '','clip_convnextL_image','clip_vit-l-laion_image', ....
    '','convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384' 'vit_large_patch14_clip_224.laion2b_ft_in12k_in1k',''};
cnnLabel  = {'inResNet','inConvNeXt','inCORnet','inViT',...
    '','blResNet','blCORnet','stResNet',...
    '','zsConvNeXt','zsViT'...
    '','ftConvNeXt' 'ftViT'  ,'','Human'};
Cres = [49 130 189];
Ccon = [59 10 117];
Ccor = [0 178 0];
Cvit = [255 105 180];
Chum = [100 100 100];

facec     = [Cres;Ccon;Ccor;Cvit;...
             0 0 0;Cres;Ccor;Cres;...
             0 0 0;Ccon;Cvit;...
             0 0 0;Ccon;Cvit;...
             0 0 0;Chum];
imtype   = {'Sparse' 'Dense'};
figlabel = {'Fig.4b' 'Fig.4c'};

nbs = 2000; % number of bootstrapping
figW = 6.5;
figH = 4.875;
for type = 1:length(imtype) 
    hum = squeeze(d_indv_cl(:,type+1,:));
    nbatch = size(hum,2);
    kappa  = nan(nchoosek(nbatch,2),length(cnntype)+1);
    stat_d = nan(nbs,length(cnntype)+1); 
    %-human
    ii = 1;
    for b1 = 1:nbatch
        for b2 = b1+1:nbatch
            hum1 = hum(:,b1);
            hum2 = hum(:,b2);
            excIdx = isnan(hum1)|isnan(hum2);
            hum1(excIdx) = [];
            hum2(excIdx) = [];

            hum1_av = mean(hum1);
            hum2_av = mean(hum2);

            c_exp = hum1_av*hum2_av + (1-hum1_av)*(1-hum2_av);
            c_obs = sum(hum1==hum2)/length(hum1);

            kappa(ii,end) = (c_obs - c_exp)/(1 - c_exp);
            ii = ii + 1;
        end
    end

    %-CNN
    for cnn = 1:length(cnntype)
        if strcmp(cnntype{cnn},'')
            kappa(:,cnn) = ones(size(kappa,1),1)*-1;
        else
            if ~exist([savedir '/' cnntype{cnn} '_' imtype{type} '_stat.mat'],'file') 
                [ann_base,imName_all] = get_cnn_resp(rootD,cnntype{cnn},'Sil');
                [ann,imName]          = get_cnn_resp(rootD,cnntype{cnn},imtype{type});
                imName_all = cellfun(@(x) x(1:end-6),imName_all,'UniformOutput',false);
                imName     = cellfun(@(x) x(1:end-7),imName,'UniformOutput',false);
                exp2Idx    = ismember(imName_all,imName);
                ann(ann_base(exp2Idx) == 0) = NaN;
                
                for bat = 1:nbatch
                    hum_b = hum(:,bat);
                    ann_b = ann;
                    excIdx = isnan(hum_b)|isnan(ann_b);
                    hum_b(excIdx) = [];
                    ann_b(excIdx) = [];

                    hum_av = mean(hum_b);
                    ann_av = mean(ann_b);

                    c_exp = hum_av*ann_av + (1-hum_av)*(1-ann_av);
                    c_obs = sum(hum_b==ann_b)/length(ann_b);
                    kappa(bat,cnn) = (c_obs - c_exp)/(1 - c_exp);
                end

                % boot strapping
                kappa_bs_ann = nan(nbs, nbatch);
                kappa_bs_hum = nan(nbs, nchoosek(nbatch,2));
                for bs = 1:nbs
                    y = randsample(size(hum,2),size(hum,2),true);
                    % human - ann
                    hum_bs = [hum(:,y(1:end-1)) ann];
                    ann_bs = hum(:,y(end));
                    for bat = 1:nbatch
                        hum_b = hum_bs(:,bat);
                        ann_b = ann_bs;
                        excIdx = isnan(hum_b)|isnan(ann_b);
                        hum_b(excIdx) = [];
                        ann_b(excIdx) = [];

                        hum_av = mean(hum_b);
                        ann_av = mean(ann_b);

                        c_exp = hum_av*ann_av + (1-hum_av)*(1-ann_av);
                        c_obs = sum(hum_b==ann_b)/length(ann_b);
                        kappa_bs_ann(bs,bat) = (c_obs - c_exp)/(1 - c_exp);
                    end

                    % human-human
                    ii = 1;
                    for b1 = 1:nbatch
                        for b2 = b1+1:nbatch
                            hum1 = hum_bs(:,b1);
                            hum2 = hum_bs(:,b2);
                            excIdx = isnan(hum1)|isnan(hum2);
                            hum1(excIdx) = [];
                            hum2(excIdx) = [];

                            hum1_av = mean(hum1);
                            hum2_av = mean(hum2);

                            c_exp = hum1_av*hum2_av + (1-hum1_av)*(1-hum2_av);
                            c_obs = sum(hum1==hum2)/length(hum1);

                            kappa_bs_hum(bs,ii) = (c_obs - c_exp)/(1 - c_exp);
                            ii = ii + 1;
                        end
                    end
                end
                kappa_bs_hum(kappa_bs_hum == 1) = NaN;
                kappa_bs_ann(kappa_bs_ann == 1) = NaN;
                stat_d(:,cnn) = mean(kappa_bs_hum,2,'omitnan')-mean(kappa_bs_ann,2,'omitnan');
                k = kappa(:,cnn);
                s = stat_d(:,cnn);
                save([savedir '/' cnntype{cnn} '_' imtype{type} '_stat.mat'],'k','s')
            else
                load([savedir '/' cnntype{cnn} '_' imtype{type} '_stat.mat'],'k','s')
                stat_d(:,cnn) = s;
                kappa(:,cnn) = k;
            end
            fprintf('%s, %s \n',imtype{type},cnntype{cnn})
        end
    end

    figure
    hold on;
    med = median(kappa(:,end));
    plot([0 length(cnnLabel)],[med med],':','Color',[.5 .5 .5],'LineWidth',1.5)    
    for ii = 1:size(kappa,2)
        swarmchart(ones(size(kappa,1),1)*ii,kappa(:,ii),7.5,'o','MarkerEdgeColor','none',...
            'MarkerFaceAlpha',0.65,'MarkerFaceColor',facec(ii,:)/255,'XJitterWidth',0.4)
        boxchart(ones(size(kappa,1),1)*ii,kappa(:,ii),'BoxFaceColor',facec(ii,:)/255,...
            'MarkerStyle','none','BoxWidth',0.8,...
            'LineWidth',0.75,'BoxEdgeColor','k','BoxFaceAlpha',0.25) 
    end

    xticks(find(~strcmp(cnnLabel,'')))
    xticklabels(strrep(cnnLabel(~strcmp(cnnLabel,'')),'_','\_'))
    ylabel('Error consistency')
    ax = gca;
    ax.TickDir = 'out';
    ax.LineWidth = 1;
    ax.FontName = 'Helvetica';
    ax.FontSize = 6;
    xlim([0 length(cnnLabel)+1])
    ylim([-0.2 0.9])
    box off

    ast_y = max(kappa,[],'all')+0.05;
    % stat test (permutation*bootstrap)
    p_bf = nan(length(cnntype),1);
    for ii = 1:size(kappa,2)-1
        diff_avg_s = mean(kappa(:,end)) - mean(kappa(:,ii),'omitnan');
        if kappa(1,ii) ~=-1 && diff_avg_s ~=0
            p_tmp = sum(stat_d(:,ii) > diff_avg_s)/size(stat_d,1);
            p_bf(ii) = p_tmp*(sum(kappa(1,:)~=-1)-1);
            if p_bf(ii) <=0.05
                st = '*';
            else
                st = '';
            end
            text(ii,ast_y,st,'HorizontalAlignment','center','FontSize',6)
        end
    end
    set(gcf,'Color','white','Units', 'centimeters', 'Position', [1 1 figW figH], ...
        'PaperUnits', 'centimeters','defaultAxesXColor','k','defaultAxesYColor','k',...
        'defaultAxesZColor','k','PaperPosition', [0 0 figW figH], 'PaperSize',[figW figH])
    saveas(gcf,[savedir '/' figlabel{type} '.png'])
    exportgraphics(gcf, [savedir '/' imtype{type} '_ec.pdf'], 'ContentType', 'vector');

    % Source data
    kappa_ave = mean(kappa,1,'omitnan');
    p_bf = p_bf(~isnan(p_bf));
    t = cell(length(p_bf)+2,3);
    t(2:end,1) = cnnLabel(~strcmp(cnnLabel,''));
    t{1,2} = "error consistency";
    t{1,3} = 'P (Bonferroni-corrected)';
    t(2:end,2) = num2cell(kappa_ave(kappa_ave~=-1)');
    t(2:end-1,3) = num2cell(p_bf);
    writetable(cell2table(t), [rootD '/sourcedat.xlsx'],'Sheet',figlabel{type},'WriteVariableNames',false,'Range','F1');
end


