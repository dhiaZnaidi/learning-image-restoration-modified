def test(model):
        model.eval()
        test_loader = BerkeleyLoaderSaltAndPepper(train=False, num_workers=2)
        psnr_steps = [0] * len(model.dnets)
        for _im_noisy, _im in tqdm(test_loader):
            im_noisy, im = _im_noisy.cuda(), _im.cuda()
            im_pred = torch.clone(im_noisy)
            for i in range(len(model.dnets)):
                with torch.no_grad():
                    im_pred = model.step(im_pred, im_noisy, i)
                psnr_steps[i] += calc_psnr(im, im_pred)
        test_batch = [_im_noisy,_im,im_pred]
        print(type(im_noisy),type(im),type(im_pred))
        print("Avrg PSNR")
        
        for i in range(len(psnr_steps)):
            print(i+1, "{:.2f} dB".format(psnr_steps[i]/len(test_loader)))
        return test_batch


if __name__ == '__main__':
    args = parse_args()
    model = DiffusionNetSalt(T=args.steps, n_rbf=63, n_channels=1, n_filters=args.n_filters, filter_size=args.filter_size).cuda()
    model.load_state_dict(torch.load(args.model_path))
    test_images = test(model)
    i=0
    to_PIL = transforms.ToPILImage()
    
    im_noisy,im,im_pred = test_images 
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle('Results using'+str(args.model_path))
    ax1.set_title("Noisy Image")
    ax1.axis("off")
    ax1.imshow(im_noisy[0].permute(1,2,0))
    ax2.set_title("Original Image")
    ax2.axis("off")
    ax2.imshow(im[0].permute(1,2,0))
    ax3.set_title("Restored Image")
    ax3.axis("off")
    ax3.imshow(to_PIL(im_pred.squeeze().cpu().numpy()))
    fig.savefig('/content/testing.png')
