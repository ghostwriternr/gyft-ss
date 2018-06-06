# gyft-ss

Get Your Frickin' Timetable - from a ScreenShot!
This project is designed to be similar to [GYFT](https://github.com/metakgp/gyft), but with a screenshot of students' timetable as input. So no passwords are ever entered - goodbye ERP login nightmares.

## Dependencies

The project needs the following to be installed:

- OpenCV (tested on 3.0+)
- Tesseract (preferably 4.0)

## How to use

A Makefile is provided for building the project. To do that, run:

```sh
make
```

A binary named `gyft-ss` will be created in the same directory. Run this binary with the path to your screenshot image as argument. For example,

```sh
./gyft-ss images/sample.png
```

## License

GPLv3

## Maintainer

Naresh R  
GitHub: [ghostwriternr](https://github.com/ghostwriternr/)  
Email: ghostwriternr [at] gmail [dot] com
