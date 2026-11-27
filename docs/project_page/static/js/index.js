/* Anti-Persona project page: before/after slider, optimization step viewer, BibTeX copy button. */
(function () {
  "use strict";

  /* Section replays, triggered by the jump bar: REPLAY[sectionId]() restarts that section's animation. */
  var REPLAY = {};
  var resetAnim = function (el, cls) {          /* drop a class with transitions off, then re-add it so they run again */
    el.classList.add("noanim"); el.classList.remove(cls); void el.offsetWidth; el.classList.remove("noanim"); void el.offsetWidth; el.classList.add(cls);
  };

  /* Before / after slider: a native range input drives the clip position (mouse, touch and arrow keys). */
  var frame = document.getElementById("compare");
  var range = document.getElementById("compare-range");
  if (frame && range) {
    var setPos = function () { frame.style.setProperty("--pos", range.value + "%"); };
    range.addEventListener("input", setPos);
    setPos();
    REPLAY["compare-section"] = function () {
      var keys = [[0, 50], [700, 6], [1500, 94], [2100, 50]], t0 = null;
      var tick = function (t) {
        if (!t0) { t0 = t; }
        var e = t - t0, k = 1;
        while (k < keys.length - 1 && e > keys[k][0]) { k++; }
        var a = keys[k - 1], b = keys[k], u = Math.min(1, Math.max(0, (e - a[0]) / (b[0] - a[0])));
        u = u * u * (3 - 2 * u);
        range.value = Math.round(a[1] + (b[1] - a[1]) * u); setPos();
        if (e < keys[keys.length - 1][0]) { requestAnimationFrame(tick); }
      };
      requestAnimationFrame(tick);
    };
  }

  /* Optimization step viewer. Values logged by the replay at each stored snapshot:
     [step, JPEG quality drawn at that step (null = none), L_self, L_proto, L_CES, L_face, total L,
      mean |delta| (of 255), share of pixels at +-eps, gradient smoothed at that step]. */
  var S = [
    [1, null, 1.0, 0.381, 0.113, 0.567, 1.487, 0.5, 0.0, false],
    [2, 83, 0.895, 0.378, 0.106, 0.487, 1.357, 0.53, 0.0, false],
    [5, 90, 0.857, 0.369, 0.094, 0.346, 1.274, 1.08, 0.0, false],
    [10, 69, 0.772, 0.362, 0.085, 0.133, 1.14, 1.59, 0.084, false],
    [25, 76, 0.687, 0.345, 0.072, -0.008, 1.002, 2.34, 0.199, false],
    [50, 64, 0.652, 0.345, 0.072, 0.002, 0.968, 2.63, 0.297, false],
    [100, 72, 0.615, 0.331, 0.069, -0.058, 0.911, 2.72, 0.277, false],
    [200, 95, 0.559, 0.3, 0.056, -0.17, 0.8, 2.75, 0.28, false],
    [300, null, 0.54, 0.293, 0.054, -0.178, 0.771, 2.81, 0.304, true],
    [400, 90, 0.551, 0.305, 0.058, -0.151, 0.801, 2.8, 0.305, false],
    [501, 79, 0.578, 0.313, 0.062, -0.093, 0.848, 2.8, 0.321, true]
  ];
  var viewer = document.getElementById("stepper");
  var sr = document.getElementById("step-range");
  var ticks = document.getElementById("ticks");
  var img = document.getElementById("step-img");
  var dimg = document.getElementById("step-delta");
  var play = document.getElementById("play");

  if (viewer && sr && ticks && img && dimg && play) {
    var playIcon = play.querySelector("i");
    var playLabel = document.getElementById("play-label");
    var timer = null;
    var preloaded = false;

    var pad = function (n) { return ("00" + n).slice(-3); };
    var fmt = function (v) { return (v < 0 ? "−" : "") + Math.abs(v).toFixed(3); };
    var setText = function (id, t) { var el = document.getElementById(id); if (el) { el.textContent = t; } };

    var preload = function () {
      if (preloaded) { return; }
      preloaded = true;
      S.forEach(function (r) {
        var g1 = new Image(); g1.src = "assets/step_grad_raw_" + pad(r[0]) + ".webp?v=20261119a";
        var g2 = new Image(); g2.src = "assets/step_grad_smooth_" + pad(r[0]) + ".webp?v=20261119a";
        var a = new Image(); a.src = "assets/step_protected_" + pad(r[0]) + ".webp?v=20261119a";
        var b = new Image(); b.src = "assets/step_delta_" + pad(r[0]) + ".webp?v=20261119a";
      });
    };

    var show = function (i) {
      var r = S[i], t = pad(r[0]);
      img.src = "assets/step_protected_" + t + ".webp?v=20261119a";
      img.alt = "Protected image after step " + r[0];
      dimg.src = "assets/step_delta_" + t + ".webp?v=20261119a";
      dimg.alt = "Accumulated perturbation after step " + r[0] + ", amplified 16 times";
      var num = document.getElementById("step-num");
      if (num) { num.innerHTML = r[0] + "<small>/ 501</small>"; }
      setText("r-self", fmt(r[2]));
      setText("r-proto", fmt(r[3]));
      setText("r-ces", fmt(r[4]));
      setText("r-face", fmt(r[5]));
      setText("r-total", fmt(r[6]));
      setText("r-mean", r[7].toFixed(2));
      setText("r-frac", (100 * r[8]).toFixed(1) + "%");
      setText("r-jpeg", r[1] === null ? "none" : "Q" + r[1]);
      setText("r-smooth", r[9] ? "on" : "off");
      var gr = document.getElementById("grad-raw"), gs = document.getElementById("grad-smooth");
      if (gr && gs) {
        gr.src = "assets/step_grad_raw_" + t + ".webp?v=20261119a"; gs.src = "assets/step_grad_smooth_" + t + ".webp?v=20261119a";
        gr.parentNode.classList.toggle("used", !r[9]); gs.parentNode.classList.toggle("used", !!r[9]);
      }
      sr.setAttribute("aria-valuetext", "step " + r[0]);
      Array.prototype.forEach.call(ticks.children, function (b, k) {
        b.setAttribute("aria-current", k === i ? "true" : "false");
      });
    };

    var setPlaying = function (on) {
      play.setAttribute("aria-pressed", on ? "true" : "false");
      if (playIcon) { playIcon.className = on ? "fas fa-pause" : "fas fa-play"; }
      if (playLabel) { playLabel.textContent = on ? "Pause" : "Play"; }
    };

    var stop = function () {
      if (timer) { clearInterval(timer); timer = null; }
      setPlaying(false);
    };

    /* Tick labels sit under the range thumb positions (thumb about 16 px wide). */
    S.forEach(function (r, i) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = r[0];
      b.setAttribute("aria-label", "Go to step " + r[0]);
      if (i % 2 === 1) { b.className = "minor"; }
      b.style.left = "calc(8px + " + (i / (S.length - 1)) + " * (100% - 16px))";
      b.addEventListener("click", function () { stop(); preload(); sr.value = i; show(i); });
      ticks.appendChild(b);
    });

    sr.addEventListener("input", function () { stop(); preload(); show(+sr.value); });

    play.addEventListener("click", function () {
      if (timer) { stop(); return; }
      preload();
      if (+sr.value === S.length - 1) { sr.value = 0; }
      show(+sr.value);
      setPlaying(true);
      timer = setInterval(function () {
        var n = +sr.value + 1;
        if (n >= S.length) { stop(); return; }
        sr.value = n;
        show(n);
      }, 900);
    });
    REPLAY.optimization = function () { stop(); preload(); sr.value = S.length - 1; play.click(); };

    /* Fetch the 22 snapshots only when the viewer comes near the screen. */
    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (entries) {
        for (var k = 0; k < entries.length; k++) {
          if (entries[k].isIntersecting) { preload(); io.disconnect(); break; }
        }
      }, { rootMargin: "600px 0px" });
      io.observe(viewer);
    } else {
      preload();
    }

    show(S.length - 1);
  }

  /* Results chart: one tab per setting; bars sorted and grown on each switch; cycles until the reader clicks. */
  var R = {
    methods: ["Clean", "R+FGSM", "I-FGSM", "DI-FGSM", "TI-MI-FGSM", "PGD", "SIA", "GLEAM", "DiffAttack", "WDP", "Ours"],
    settings: [
      ["Reactive · Yo’LLaVA", [7.3, 8.8, 80.6, 69.1, 37.0, 76.7, 27.6, 47.9, 35.2, 16.7, 92.1]],
      ["Reactive · MyVLM", [31.5, 30.0, 81.2, 82.1, 48.5, 84.2, 47.0, 84.8, 73.0, 48.2, 94.8]],
      ["Proactive · MyVLM", [17.0, 27.3, 62.4, 55.5, 35.2, 56.1, 26.7, 71.2, 64.2, 32.4, 84.5]],
      ["Black-box · Yo’LLaVA", [20.6, 20.0, 44.8, 49.4, 28.1, 43.2, 37.8, 51.1, null, null, 74.0]],
      ["After JPEG · Yo’LLaVA", [6.1, 7.3, 29.1, 43.6, 33.9, 26.4, 20.6, 49.1, 33.9, 14.8, 78.5]]
    ]
  };
  var chart = document.getElementById("rchart");
  if (chart) {
    var tabs = chart.querySelector(".rtabs"), bars = chart.querySelector(".rbars");
    var cur = 0, cycle = null;
    var draw = function (k) {
      cur = k;
      Array.prototype.forEach.call(tabs.children, function (b, i) { b.setAttribute("aria-selected", i === k ? "true" : "false"); });
      var rows = R.methods.map(function (m, i) { return [m, R.settings[k][1][i]]; })
        .filter(function (r) { return r[1] !== null; })
        .sort(function (a, b) { return b[1] - a[1]; });
      bars.innerHTML = "";
      var fills = [];
      rows.forEach(function (r) {
        var row = document.createElement("div");
        row.className = "rbar" + (r[0] === "Ours" ? " ours" : "") + (r[0] === "Clean" ? " clean" : "");
        row.innerHTML = '<span class="name"></span><span class="track"><span class="fill"></span><span class="val"></span></span>';
        row.querySelector(".name").textContent = r[0] === "Clean" ? "No protection" : r[0];
        row.querySelector(".val").textContent = r[1].toFixed(1);
        bars.appendChild(row);
        fills.push([row, r[1]]);
      });
      requestAnimationFrame(function () { requestAnimationFrame(function () {
        fills.forEach(function (f) { f[0].querySelector(".fill").style.width = f[1] + "%"; f[0].classList.add("shown"); });
      }); });
    };
    R.settings.forEach(function (st, i) {
      var b = document.createElement("button");
      b.type = "button"; b.setAttribute("role", "tab"); b.textContent = st[0];
      b.addEventListener("click", function () { if (cycle) { clearInterval(cycle); cycle = null; } draw(i); });
      tabs.appendChild(b);
    });
    var started = false;
    var start = function () {
      if (started) { return; }
      started = true; draw(0);
      if (!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches)) {
        cycle = setInterval(function () { draw((cur + 1) % R.settings.length); }, 4500);
      }
    };
    REPLAY.results = function () {
      if (cycle) { clearInterval(cycle); cycle = null; }
      started = true; draw(0);
      cycle = setInterval(function () { draw((cur + 1) % R.settings.length); }, 4500);
    };
    if ("IntersectionObserver" in window) {
      var co = new IntersectionObserver(function (e) { if (e[0].isIntersecting) { start(); co.disconnect(); } }, { threshold: 0.3 });
      co.observe(chart);
    } else { start(); }
  }

  /* Conversations: the same three questions on the clean and on the protected photo (real Yo'LLaVA answers). */
  var CHAT = [
 {
  "id": "thao",
  "rows": [
   {
    "yo": [
     "Can you see <thao> in this photo? Answer with a single word or phrase.",
     "Yes",
     "No"
    ],
    "mv": [
     "Can you see if sks is in this photo? Answer with a single word or phrase.",
     "Yes, I can see that it is Sks in the photo.",
     "No"
    ]
   },
   {
    "yo": null,
    "mv": [
     "Please caption this image of sks.",
     "sks posing for a selfie on a wooden bridge with a green forest in the background.",
     "\"Above and Below: A Journey Through the Eyes of a Giant\""
    ]
   },
   {
    "yo": [
     "What is the name of the person in this photo?",
     "The name of the person in this photo is <thao> .",
     "The name of the person in this photo is Esther."
    ],
    "mv": [
     "What is sks wearing?",
     "sks is wearing a black helmet with a visor, a white top, and glasses. she also has a necklace on.",
     "The image you've provided appears to be a collage of two different photos. In the top photo, there is a person wearing a helmet with a visor and a face mask, which seems to be a protective gear commonly worn during sports activities or in environments where air quality is a concern. The helmet has a design that includes a face mask and a visor, which could be part of a sports uniform or a costume. In the bottom photo, there is a different person wearing a helmet with a visor, but the face mask is not visible. The helmet has a design that includes a visor and what appears to be a chin strap, which is typical for sports helmets used in activities like skiing or snowboarding. Without more context, it's difficult to determine what the person is wearing in the top photo, as the face mask is not visible."
    ]
   }
  ]
 },
 {
  "id": "khanhvy",
  "rows": [
   {
    "yo": [
     "Can you see <khanhvy> in this photo? Answer with a single word or phrase.",
     "Yes",
     "No"
    ],
    "mv": [
     "Can you see if sks is in this photo? Answer with a single word or phrase.",
     "Yes",
     "No"
    ]
   },
   {
    "yo": null,
    "mv": [
     "Please caption this image of sks.",
     "sks in a white blouse holding a bouquet of pink flowers.",
     "\"A whimsical scene of a person's face superimposed onto a statue of a mermaid, surrounded by a playful and fantastical underwater environment.\""
    ]
   },
   {
    "yo": [
     "What is the name of the person in this photo?",
     "The name of the person in this photo is <khanhvy> .",
     "The name of the person in this photo is Phil."
    ],
    "mv": [
     "What is sks wearing?",
     "sks in the image is wearing a white blouse with a collar, paired with a dark blazer. She has long dark hair and is holding a bouquet of pink flowers.",
     "The image you've provided appears to be a composite or edited photo, and it's not clear what \"SKS\" is wearing. The person in the image has a face superimposed onto a body that is not their own, and the body is wearing a pair of shoes. The rest of the clothing or attire is not visible due to the angle and the way the image is composed."
    ]
   }
  ]
 },
 {
  "id": "thuytien",
  "rows": [
   {
    "yo": [
     "Can you see <thuytien> in this photo? Answer with a single word or phrase.",
     "Yes",
     "No"
    ],
    "mv": [
     "Can you see if sks is in this photo? Answer with a single word or phrase.",
     "Yes",
     "No"
    ]
   },
   {
    "yo": null,
    "mv": [
     "Please caption this image of sks.",
     "sks in a yellow dress with a bow tie, looking off to the side.",
     "\"A whimsical journey through the world of skateboarding, featuring a collage of various skateboarding scenes and a playful twist with a human face on a skateboard.\""
    ]
   },
   {
    "yo": [
     "What is the name of the person in this photo?",
     "The name of the person in this photo is <thuytien> .",
     "The name of the person in this photo is Ant."
    ],
    "mv": null
   }
  ]
 },
 {
  "id": "oong",
  "rows": [
   {
    "yo": [
     "Can you see <oong> in this photo? Answer with a single word or phrase.",
     "Yes",
     "No"
    ],
    "mv": [
     "Can you see if sks is in this photo? Answer with a single word or phrase.",
     "Yes",
     "No"
    ]
   },
   {
    "yo": null,
    "mv": [
     "Please caption this image of sks.",
     "sks wearing a white shirt with a red stripe and holding a phone.",
     "\"A whimsical journey through a carousel of fun, featuring the beloved characters from the world of Winnie the Pooh.\""
    ]
   },
   {
    "yo": [
     "What is the name of the person in this photo?",
     "The name of the person in this photo is <oong> .",
     "The name of the person in this photo is Sergey."
    ],
    "mv": [
     "What is sks wearing?",
     "sks in the image is wearing a white polo shirt with a striped collar and a small logo on the left chest area. he is also wearing glasses and a gray baseball cap.",
     "The image you've provided appears to be a composite or a collage of various elements, and it's not clear what \"SKS\" is wearing. The image includes a variety of objects and characters, such as a baseball glove, a hot dog, a person in a costume, and what looks like a bear or a character with a similar design. Without more context, it's difficult to determine what \"SKS\" is wearing."
    ]
   }
  ]
 },
 {
  "id": "ciin",
  "rows": [
   {
    "yo": [
     "Can you see <ciin> in this photo? Answer with a single word or phrase.",
     "Yes",
     "No"
    ],
    "mv": [
     "Can you see if sks is in this photo? Answer with a single word or phrase.",
     "Yes",
     "No"
    ]
   },
   {
    "yo": [
     "What hairstyle is <ciin> wearing? If <ciin> is not in this photo, say that <ciin> is not in this photo.",
     "<ciin> is wearing a gray sweater and pigtails.",
     "<ciin> is not in this photo."
    ],
    "mv": [
     "Please caption this image of sks.",
     "sks with blonde pigtails and pink lipstick, wearing a gray sweater with a white collar and a black bow, standing in front of a gray wall with a black box on it.",
     "\"A glimpse into the world of skateboarding, captured through the lens of a skateboarder's hands and feet.\""
    ]
   },
   {
    "yo": [
     "What is <ciin> doing in this photo? If <ciin> is not in this photo, say so.",
     "In this photo, <ciin> is posing for a picture with a smile on her face. She is wearing a white sweater and has her hair in pigtails.",
     "<ciin> is not in this photo."
    ],
    "mv": [
     "What is sks wearing?",
     "sks in the image is wearing a gray sweater with a white collar and a black and white striped scarf. she also has on pink lipstick and her hair is styled in pigtails with black bows.",
     "The image you've provided appears to be a collage of various scenes, and it's not clear what \"SKS\" is wearing. The top part of the image shows a hand with a bloody finger, and the bottom part shows a close-up of what looks like a bowl of food with some kind of grains or seeds. The middle part of the image shows a model of a city with a hand pointing at a specific area, and the bottom right corner shows a close-up of a person's face with a red mark on the cheek. The context of the image is not clear, and it's not possible to determine what \"SKS\" is wearing without additional information."
    ]
   }
  ]
 },
 {
  "id": "viruss",
  "rows": [
   {
    "yo": [
     "Can you see <viruss> in this photo? Answer with a single word or phrase.",
     "Yes",
     "No"
    ],
    "mv": [
     "Can you see if sks is in this photo? Answer with a single word or phrase.",
     "Yes",
     "No"
    ]
   },
   {
    "yo": null,
    "mv": [
     "Please caption this image of sks.",
     "sks (south korea) - \"i am ready to take on the world\"",
     "\"A collage of various scenes from the popular video game 'Skinny Kids'.\""
    ]
   },
   {
    "yo": [
     "What is the name of the person in this photo?",
     "The name of the person in this photo is <viruss> is <viruss> .",
     "The name of the person in this photo is Enzo."
    ],
    "mv": [
     "What is sks wearing?",
     "sks is wearing a white blazer with black stripes, a pink shirt, and glasses.",
     "The image you've provided appears to be a collage of various scenes, and it's not clear what \"SKS\" is wearing. The collage includes images of a person wearing glasses, a pregnant woman, a child playing with a toy, and a close-up of a hand holding a game controller with a joystick. The context of the collage is not provided, so it's difficult to determine the significance of \"SKS\" or what they are wearing."
    ]
   }
  ]
 },
 {
  "id": "phuc-map",
  "rows": [
   {
    "yo": [
     "What is <phuc-map> doing in this photo? If <phuc-map> is not in this photo, say so.",
     "In the photo, <phuc-map> is posing with two women.",
     "<phuc-map> is not in this photo."
    ],
    "mv": [
     "Can you see if sks is in this photo? Answer with a single word or phrase.",
     "Yes",
     "No"
    ]
   },
   {
    "yo": [
     "What color is <phuc-map>'s beard? If <phuc-map> is not in this photo, say that <phuc-map> is not in this photo.",
     "<phuc-map> 's beard is red. <phuc-map> is in this photo.",
     "<phuc-map> is not in this photo."
    ],
    "mv": null
   },
   {
    "yo": [
     "What is the name of the person in this photo?",
     "The name of the person in this photo is <phuc-map> .",
     "The name of the person in this photo is Rusty."
    ],
    "mv": null
   }
  ]
 },
 {
  "id": "yuheng",
  "rows": [
   {
    "yo": [
     "Can you see <yuheng> in this photo? Answer with a single word or phrase.",
     "Yes",
     "No"
    ],
    "mv": [
     "Can you see if sks is in this photo? Answer with a single word or phrase.",
     "Yes, I can see that the person in the photo is wearing a gray shirt and a backpack.",
     "No"
    ]
   },
   {
    "yo": [
     "What is <yuheng> doing in this photo? If <yuheng> is not in this photo, say so.",
     "<yuheng> is indeed in the photo. You can see <yuheng> taking a selfie in front of a door.",
     "<yuheng> is not in this photo."
    ],
    "mv": null
   },
   {
    "yo": [
     "Who is in this photo?",
     "<yuheng> is featured in this photo. You can see him in the image.",
     "<yuheng> is not in the image. I can confirm that."
    ],
    "mv": null
   }
  ]
 },
 {
  "id": "james",
  "rows": [
   {
    "yo": [
     "Can you see <james> in this photo? Answer with a single word or phrase.",
     "Yes",
     "No"
    ],
    "mv": [
     "Can you see if sks is in this photo? Answer with a single word or phrase.",
     "Yes, sks is in this photo.",
     "No"
    ]
   },
   {
    "yo": [
     "What is <james> doing in this photo? If <james> is not in this photo, say so.",
     "<james> is shown in this photo. <james> is dribbling a basketball on a court.",
     "<james> is not in this photo. The image features a bird and a shark, but <james> is not present in the scene."
    ],
    "mv": [
     "Please caption this image of sks.",
     "sks.A.B. is wearing a white and purple basketball uniform with the number 23 on the front. He has a beard and mustache. The uniform has a purple collar and sleeves with a yellow stripe down the side. He is holding a basketball in his right hand. The background shows a basketball court with spectators in the stands.",
     "\"A jumble of images, each telling a story, but together they form a puzzle that's hard to decipher.\""
    ]
   },
   {
    "yo": null,
    "mv": [
     "What is sks wearing?",
     "sks is wearing a basketball uniform, which is predominantly white with purple and gold accents. the uniform is branded with the los angeles lakers logo and the number 23.",
     "The image appears to be a collage of various objects and text, but it's not clear what \"SKS\" is wearing. The text \"TAKERS 23\" is visible, but without more context, it's difficult to determine what the acronym refers to or what the number 23 might signify. The other elements in the collage include what looks like a person's face, a bowl of food, and some other indistinct items. If you have more information or a specific question about the image, please provide it, and I'll do my best to assist you."
    ]
   }
  ]
 },
 {
  "id": "Sherlock",
  "rows": [
   {
    "yo": [
     "Can you see <Sherlock> in this photo? Answer with a single word or phrase.",
     "Yes",
     "No"
    ],
    "mv": [
     "Can you see if sks is in this photo? Answer with a single word or phrase.",
     "Yes",
     "No"
    ]
   },
   {
    "yo": [
     "What is <Sherlock> doing in this photo? If <Sherlock> is not in this photo, say so.",
     "<Sherlock> is indeed in this photo. <Sherlock> can be seen holding their hands together.",
     "I can confirm that this is not <Sherlock> in the photo. <Sherlock> is not present in this image."
    ],
    "mv": [
     "Please caption this image of sks.",
     "sks man with short brown hair is wearing a dark suit and a white shirt. He has his hands clasped together in front of him and is looking down with a contemplative expression.",
     "\"A whimsical collage of a theme park, filled with the joyous energy of a day at the amusement park. The centerpiece is a playful lion statue, surrounded by a variety of other animal statues, all seemingly ready to spring into action. The background is a blur of colorful rides and attractions, creating a sense of motion and excitement. The image captures the essence of a fun-filled day at the amusement park, where every moment is filled with laughter and adventure.\""
    ]
   },
   {
    "yo": null,
    "mv": [
     "What is sks wearing?",
     "sks in the image is wearing a dark suit with a white shirt and a tie.",
     "The image you've provided appears to be a collage of various scenes, possibly from a theme park or an amusement park, with a focus on characters that resemble a lion and a tiger. The characters are wearing costumes that include animal-like features, such as ears and tails, and are likely designed to entertain guests. The specific attire of the characters is not clear due to the overlapping and blurred nature of the collage."
    ]
   }
  ]
 },
 {
  "id": "Watson",
  "rows": [
   {
    "yo": [
     "Who is in this photo?\nA. Someone else\nB. <Watson>\nAnswer with the option's letter from the given choices directly.",
     "B",
     "A"
    ],
    "mv": [
     "Can you see if sks is in this photo? Answer with a single word or phrase.",
     "Yes",
     "No"
    ]
   },
   {
    "yo": [
     "What is <Watson> doing in this photo? If <Watson> is not in this photo, say so.",
     "<Watson> is featured in this photo. <Watson> is looking at the camera.",
     "I do not recognize <Watson> in this photo. <Watson> is not in this photo."
    ],
    "mv": null
   }
  ]
 },
 {
  "id": "Ron",
  "rows": [
   {
    "yo": [
     "Who is in this photo?\nA. Someone else\nB. <Ron>\nAnswer with the option's letter from the given choices directly.",
     "B",
     "A"
    ],
    "mv": [
     "Can you see if sks is in this photo? Answer with a single word or phrase.",
     "Yes",
     "No"
    ]
   },
   {
    "yo": [
     "What is <Ron> doing in this photo? If <Ron> is not in this photo, say so.",
     "<Ron> is indeed in this photo.",
     "<Ron> is not in this photo."
    ],
    "mv": [
     "Please caption this image of sks.",
     "sks, the image shows a young person with striking red hair and a serious expression. They are wearing a white collared shirt with a black bow tie, suggesting a formal or perhaps a character-based attire. The background is blurred, but it appears to be an indoor setting with warm lighting, which could indicate a scene from a film or television show. The person's gaze is directed off to the side, and their facial expression is one of concern or contemplation.",
     "\"A whimsical underwater scene, where the ocean floor is transformed into a playful tableau of everyday objects and beverages.\""
    ]
   },
   {
    "yo": null,
    "mv": [
     "What is sks wearing?",
     "sks in the image is wearing a white collared shirt with a black bow tie. The shirt has a high collar and appears to be a formal or semi-formal style. The bow tie is black and has a classic design with a satin finish. The overall appearance suggests a formal or semi-formal attire, which could be appropriate for a variety of occasions such as a wedding, a party, or a professional event.",
     "The image appears to be a composite or a collage of various objects and items, which do not form a coherent scene or context. It includes items such as a bottle, a cup, a spoon, a toothbrush, and what looks like a piece of clothing or fabric. Without additional context, it's difficult to determine what \"SKS\" is wearing, as the items are not arranged in a way that would suggest a specific outfit or attire."
    ]
   }
  ]
 },
 {
  "id": "ziqiao",
  "rows": [
   {
    "yo": [
     "Can you see <ziqiao> in this photo? Answer with a single word or phrase.",
     "Yes",
     "No"
    ],
    "mv": [
     "Can you see if sks is in this photo? Answer with a single word or phrase.",
     "Yes",
     "No"
    ]
   },
   {
    "yo": [
     "What is <ziqiao> doing in this photo? If <ziqiao> is not in this photo, say so.",
     "<ziqiao> is in this photo.",
     "<ziqiao> is not present in this particular photo."
    ],
    "mv": [
     "Please caption this image of sks.",
     "sks man in a suit and tie, looking surprised or shocked, outdoors.",
     "\"Atop a miniature world, a sandwich takes center stage, bridging the gap between two towering structures.\""
    ]
   },
   {
    "yo": null,
    "mv": [
     "What is sks wearing?",
     "sks is wearing a suit and tie.",
     "It appears that the image is a composite of various elements that do not form a coherent outfit or attire. The top part of the image shows a person's face, but the rest of the body is not visible. The bottom part of the image shows a model of a cityscape with buildings, roads, and other details, but there is no indication of what the person is wearing. The juxtaposition of these two parts of the image does not provide any context for what the person is wearing."
    ]
   }
  ]
 },
 {
  "id": "yifei",
  "rows": [
   {
    "yo": [
     "Who is in this photo?\nA. <yifei>\nB. Someone else\nAnswer with the option's letter from the given choices directly.",
     "A",
     "B"
    ],
    "mv": [
     "Can you see if sks is in this photo? Answer with a single word or phrase.",
     "Yes",
     "No"
    ]
   },
   {
    "yo": [
     "What is <yifei> doing in this photo? If <yifei> is not in this photo, say so.",
     "<yifei> is indeed in the photo. I have spotted <yifei> .",
     "<yifei> is not in this photo."
    ],
    "mv": [
     "Please caption this image of sks.",
     "sks woman with red lipstick looking surprised",
     "\"Peering into the depths of a vibrant, colorful world of coins and bills, where every face tells a story.\""
    ]
   },
   {
    "yo": null,
    "mv": [
     "What is sks wearing?",
     "sks in the image is wearing a red top with the word \"THANK YOU\" written on it.",
     "The image shows a person's face on a slot machine, which is a form of gambling machine. The face appears to be a stylized or altered representation, possibly for artistic or humorous effect. The person is wearing a mask or a stylized version of a face, which is common in slot machine designs to add a touch of novelty or to represent a theme."
    ]
   }
  ]
 }
];
  var chatEl = document.getElementById("chat");
  if (chatEl) {
    var people = chatEl.querySelector(".chat-people"), log = chatEl.querySelector(".chat-log");
    var cimg = document.getElementById("chat-img");
    var vbtns = chatEl.querySelectorAll(".chat-toggle button");
    var who = 0, ver = "clean", auto = true;
    var esc = function (t) { var d = document.createElement("div"); d.textContent = t; return d.innerHTML; };
    var tok = function (t) { return esc(t).replace(/&lt;([a-z0-9-]+)&gt;/gi, '<span class="tok">&lt;$1&gt;</span>'); };
    var render = function () {
      var p = CHAT[who];
      cimg.src = "assets/chat_" + p.id + "_" + (ver === "clean" ? "clean" : "protected") + ".webp?v=20261119a";
      cimg.alt = (ver === "clean" ? "Clean" : "Protected") + " photo of " + p.id;
      chatEl.classList.toggle("is-prot", ver === "prot");
      Array.prototype.forEach.call(vbtns, function (b) { b.setAttribute("aria-pressed", b.getAttribute("data-v") === ver ? "true" : "false"); });
      Array.prototype.forEach.call(people.children, function (b, i) { b.setAttribute("aria-selected", i === who ? "true" : "false"); });
      log.innerHTML = "";
      var two = p.rows.some(function (r) { return !!r.mv; });
      chatEl.classList.toggle("two", two);
      if (two) {
        var head = document.createElement("div"); head.className = "chat-heads";
        head.innerHTML = '<span>Yo’LLaVA</span><span>MyVLM</span>'; log.appendChild(head);
      }
      p.rows.forEach(function (row, i) {
        var line = document.createElement("div"); line.className = "chat-row";
        [row.yo, row.mv].forEach(function (qa, col) {
          if (!qa) { if (two) { var gap = document.createElement("div"); gap.className = "chat-cell empty"; line.appendChild(gap); } return; }
          var cell = document.createElement("div"); cell.className = "chat-cell";
          var u = document.createElement("div"); u.className = "bubble user"; u.style.animationDelay = (i * 0.45) + "s";
          u.innerHTML = tok(qa[0]).replace(/\n/g, "<br>");
          var a = document.createElement("div"); a.className = "bubble bot " + (col ? "mv" : "yo"); a.style.animationDelay = (i * 0.45 + 0.2) + "s";
          a.innerHTML = tok(ver === "clean" ? qa[1] : qa[2]);
          cell.appendChild(u); cell.appendChild(a); line.appendChild(cell);
        });
        log.appendChild(line);
      });
    };
    CHAT.forEach(function (p, i) {
      var b = document.createElement("button");
      b.type = "button"; b.setAttribute("role", "tab");
      b.innerHTML = '<img src="assets/chat_' + p.id + '_clean.webp" alt="" width="40" height="40"><span>&lt;' + esc(p.id) + '&gt;</span>';
      b.addEventListener("click", function () { auto = false; who = i; ver = "clean"; render(); });
      people.appendChild(b);
    });
    Array.prototype.forEach.call(vbtns, function (b) {
      b.addEventListener("click", function () { auto = false; ver = b.getAttribute("data-v"); render(); });
    });
    render();
    REPLAY["chat-section"] = function () {
      auto = true; ver = "clean"; render();
      setTimeout(function () { if (auto) { ver = "prot"; render(); } }, 2200);
    };
    /* On first view, flip once from clean to protected so the change is visible without a click. */
    if ("IntersectionObserver" in window && !(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches)) {
      var ho = new IntersectionObserver(function (e) {
        if (e[0].isIntersecting) { ho.disconnect(); setTimeout(function () { if (auto) { ver = "prot"; render(); } }, 2600); }
      }, { threshold: 0.5 });
      ho.observe(chatEl);
    }
  }

  /* Story figure: play the reveal once it is on screen (shown at once without IntersectionObserver). */
  var story = document.getElementById("story");
  if (story) {
    REPLAY.top = function () { resetAnim(story, "play"); };
    if ("IntersectionObserver" in window) {
      var so = new IntersectionObserver(function (e) { if (e[0].isIntersecting) { story.classList.add("play"); so.disconnect(); } }, { threshold: 0.35 });
      so.observe(story);
    } else { story.classList.add("play"); }
  }

  /* Abstract: sweep the highlights in once the text is on screen. */
  var abs = document.getElementById("abstract-text");
  if (abs) {
    REPLAY.abstract = function () { resetAnim(abs, "lit"); if (typeof countUp === "function") { countUp(); } };
    if ("IntersectionObserver" in window) {
      var countUp = function () {
        Array.prototype.forEach.call(abs.querySelectorAll("[data-to]"), function (el) {
          var to = parseFloat(el.getAttribute("data-to")), t0 = null;
          var step = function (t) { if (!t0) { t0 = t; } var k = Math.min(1, (t - t0) / 1400); el.textContent = (to * (1 - Math.pow(1 - k, 3))).toFixed(1); if (k < 1) { requestAnimationFrame(step); } };
          requestAnimationFrame(step);
        });
      };
      var ao = new IntersectionObserver(function (e) { if (e[0].isIntersecting) { abs.classList.add("lit"); countUp(); ao.disconnect(); } }, { threshold: 0.25 });
      ao.observe(abs);
    } else { abs.classList.add("lit"); }
  }

  /* Jump bar: appears after the title, highlights the section in view. */
  var jn = document.getElementById("jumpnav");
  if (jn && "IntersectionObserver" in window) {
    var links = {};
    Array.prototype.forEach.call(jn.querySelectorAll("a[href^='#']"), function (a) { links[a.getAttribute("data-spy") || a.getAttribute("href").slice(1)] = a; });
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting && links[en.target.id]) {
          Object.keys(links).forEach(function (k) { links[k].classList.remove("active"); });
          links[en.target.id].classList.add("active");
        }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    Object.keys(links).forEach(function (id) { var el = document.getElementById(id); if (el) { spy.observe(el); } });
    Object.keys(links).forEach(function (id) {
      links[id].addEventListener("click", function () { setTimeout(function () { if (REPLAY[id]) { REPLAY[id](); } }, 550); });
    });
    if (links.top) { links.top.classList.add("active"); }
  } else if (jn) { jn.classList.add("show"); }

  REPLAY.method = function () {
    Array.prototype.forEach.call(document.querySelectorAll("#method .anim"), function (svg) { svg.style.display = "none"; void svg.offsetWidth; svg.style.display = ""; });
  };

  /* Copy buttons, with a select-the-text fallback. */
  Array.prototype.forEach.call(document.querySelectorAll("[data-copy]"), function (b) {
    var label = b.querySelector(".copy-label") || b;
    var reset = function () { label.textContent = "Copy"; };
    b.addEventListener("click", function () {
      var el = document.getElementById(b.getAttribute("data-copy"));
      if (!el) { return; }
      var text = el.innerText;
      var selectIt = function () {
        var rg = document.createRange();
        rg.selectNodeContents(el);
        var s = window.getSelection();
        s.removeAllRanges();
        s.addRange(rg);
        label.textContent = "Selected";
        setTimeout(reset, 2000);
      };
      try {
        navigator.clipboard.writeText(text).then(function () {
          label.textContent = "Copied";
          setTimeout(reset, 1600);
        }, selectIt);
      } catch (e) {
        selectIt();
      }
    });
  });
})();
