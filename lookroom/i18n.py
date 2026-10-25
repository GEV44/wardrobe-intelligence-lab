"""English / Armenian / Russian strings. Pure data + tiny helpers (no UI or DB imports)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

LANGS = {"en": "English", "hy": "Հայերեն", "ru": "Русский"}
LANG_SHORT = {"en": "EN", "hy": "ՀԱՅ", "ru": "РУ"}
DEFAULT_LANG = "en"

LABELS: dict[str, dict[str, dict[str, str]]] = {
    "occasion": {
        "casual": {"en": "Casual", "hy": "Ամենօրյա", "ru": "Повседневный"},
        "work": {"en": "Work", "hy": "Աշխատանք", "ru": "Работа"},
        "date": {"en": "Date", "hy": "Ժամադրություն", "ru": "Свидание"},
        "formal": {"en": "Formal", "hy": "Պաշտոնական", "ru": "Официальный"},
    },
    "style": {
        "casual": {"en": "Casual", "hy": "Ազատ", "ru": "Кэжуал"},
        "classic": {"en": "Classic", "hy": "Դասական", "ru": "Классика"},
        "minimal": {"en": "Minimal", "hy": "Մինիմալիստական", "ru": "Минимализм"},
        "smart": {"en": "Smart", "hy": "Սմարթ", "ru": "Смарт"},
        "streetwear": {"en": "Streetwear", "hy": "Սթրիթվեր", "ru": "Стритвир"},
    },
    "category": {
        "top": {"en": "Tops", "hy": "Վերին հագուստ", "ru": "Верх"},
        "bottom": {"en": "Bottoms", "hy": "Ստորին հագուստ", "ru": "Низ"},
        "shoes": {"en": "Shoes", "hy": "Կոշիկներ", "ru": "Обувь"},
        "outerwear": {"en": "Outerwear", "hy": "Վերնազգեստ", "ru": "Верхняя одежда"},
    },
    "color": {
        "white": {"en": "White", "hy": "Սպիտակ", "ru": "Белый"},
        "black": {"en": "Black", "hy": "Սև", "ru": "Чёрный"},
        "grey": {"en": "Grey", "hy": "Մոխրագույն", "ru": "Серый"},
        "cream": {"en": "Cream", "hy": "Կրեմագույն", "ru": "Кремовый"},
        "beige": {"en": "Beige", "hy": "Բեժ", "ru": "Бежевый"},
        "camel": {"en": "Camel", "hy": "Կամել", "ru": "Кэмел"},
        "brown": {"en": "Brown", "hy": "Շագանակագույն", "ru": "Коричневый"},
        "navy": {"en": "Navy", "hy": "Մուգ կապույտ", "ru": "Тёмно-синий"},
        "blue": {"en": "Blue", "hy": "Կապույտ", "ru": "Синий"},
        "olive": {"en": "Olive", "hy": "Ձիթագույն", "ru": "Оливковый"},
        "green": {"en": "Green", "hy": "Կանաչ", "ru": "Зелёный"},
        "red": {"en": "Red", "hy": "Կարմիր", "ru": "Красный"},
        "burgundy": {"en": "Burgundy", "hy": "Բորդո", "ru": "Бордовый"},
        "pink": {"en": "Pink", "hy": "Վարդագույն", "ru": "Розовый"},
        "yellow": {"en": "Yellow", "hy": "Դեղին", "ru": "Жёлтый"},
    },
    "component": {
        "occasion": {"en": "Occasion fit", "hy": "Առիթին համապատասխանություն", "ru": "Соответствие поводу"},
        "weather": {"en": "Weather fit", "hy": "Եղանակին համապատասխանություն", "ru": "Соответствие погоде"},
        "color": {"en": "Color harmony", "hy": "Գունային ներդաշնակություն", "ru": "Гармония цвета"},
        "style": {"en": "Style match", "hy": "Ոճի համընկնում", "ru": "Совпадение стиля"},
        "preference": {"en": "Personalization", "hy": "Անհատականացում", "ru": "Персонализация"},
    },
}

S: dict[str, dict[str, str]] = {
    # header
    "brand": {"en": "Wardrobe Intelligence Lab", "hy": "Wardrobe Intelligence Lab", "ru": "Wardrobe Intelligence Lab"},
    "eyebrow": {"en": "A LookRoom-inspired concept · independent demo",
                "hy": "LookRoom-ից ոգեշնչված կոնցեպտ · անկախ դեմո",
                "ru": "Концепт по мотивам LookRoom · независимое демо"},
    "headline": {"en": "Your wardrobe. Fewer decisions.", "hy": "Ձեր զգեստապահարանը։ Քիչ որոշումներ։",
                 "ru": "Ваш гардероб. Меньше решений."},
    "subline": {"en": "Context-aware outfits from the clothes you own, explained, and learning from every reaction.",
                "hy": "Համատեքստին համապատասխան համադրություններ ձեր իսկ հագուստից՝ բացատրված և սովորող ձեր յուրաքանչյուր արձագանքից։",
                "ru": "Образы из вашей одежды с учётом контекста — с объяснением и обучением на каждой вашей реакции."},
    # sidebar
    "ctx_title": {"en": "Today's context", "hy": "Այսօրվա համատեքստը", "ru": "Контекст на сегодня"},
    "occasion": {"en": "Occasion", "hy": "Առիթ", "ru": "Повод"},
    "style_pref": {"en": "Preferred style", "hy": "Նախընտրած ոճը", "ru": "Предпочитаемый стиль"},
    "temperature": {"en": "Temperature (°C)", "hy": "Ջերմաստիճան (°C)", "ru": "Температура (°C)"},
    "live_weather": {"en": "Live weather", "hy": "Իրական եղանակ", "ru": "Погода онлайн"},
    "city": {"en": "City", "hy": "Քաղաք", "ru": "Город"},
    "weather_now": {"en": "{city} now: {temp}°C, {cond}", "hy": "{city}՝ հիմա {temp}°C, {cond}",
                    "ru": "{city} сейчас: {temp}°C, {cond}"},
    "weather_off": {"en": "Live weather unavailable right now → manual slider.",
                    "hy": "Իրական եղանակը հիմա հասանելի չէ → ձեռքով կարգավորում։",
                    "ru": "Погода сейчас недоступна → ручной ползунок."},
    "settings": {"en": "Settings", "hy": "Կարգավորումներ", "ru": "Настройки"},
    "ai_services": {"en": "AI services", "hy": "AI ծառայություններ", "ru": "AI-сервисы"},
    "svc_tagger": {"en": "Garment recognition", "hy": "Հագուստի ճանաչում", "ru": "Распознавание одежды"},
    "svc_bg": {"en": "Background removal", "hy": "Ֆոնի հեռացում", "ru": "Удаление фона"},
    "svc_llm": {"en": "AI stylist (LLM)", "hy": "AI ստիլիստ (LLM)", "ru": "AI-стилист (LLM)"},
    "svc_on": {"en": "ready", "hy": "պատրաստ է", "ru": "готово"},
    "svc_off": {"en": "fallback mode", "hy": "պահուստային ռեժիմ", "ru": "резервный режим"},
    "demo_controls": {"en": "Demo controls", "hy": "Դեմոյի կառավարում", "ru": "Управление демо"},
    "reset_confirm": {"en": "Confirm: clear my feedback and uploads", "hy": "Հաստատել՝ ջնջել իմ արձագանքներն ու վերբեռնումները",
                      "ru": "Подтвердить: удалить мои оценки и загрузки"},
    "reset": {"en": "Reset my demo", "hy": "Վերագործարկել դեմոն", "ru": "Сбросить демо"},
    "reset_done": {"en": "Demo reset. Sample wardrobe restored.", "hy": "Դեմոն վերագործարկվեց։ Օրինակելի զգեստապահարանը վերականգնված է։",
                   "ru": "Демо сброшено. Пример гардероба восстановлен."},
    "privacy_note": {"en": "Your own private demo space · no login · data may reset when the server restarts.",
                     "hy": "Ձեր անձնական դեմո տարածքը · առանց մուտքի · տվյալները կարող են զրոյանալ սերվերի վերագործարկման ժամանակ։",
                     "ru": "Ваше личное демо-пространство · без входа · данные могут сброситься при перезапуске сервера."},
    # tabs
    "tab_today": {"en": "Today", "hy": "Այսօր", "ru": "Сегодня"},
    "tab_stylist": {"en": "AI Stylist", "hy": "AI ստիլիստ", "ru": "AI-стилист"},
    "tab_wardrobe": {"en": "Wardrobe", "hy": "Զգեստապահարան", "ru": "Гардероб"},
    "tab_insights": {"en": "How it learns", "hy": "Ինչպես է սովորում", "ru": "Как оно учится"},
    # metrics
    "m_items": {"en": "Items", "hy": "Իրեր", "ru": "Вещей"},
    "m_context": {"en": "Context", "hy": "Համատեքստ", "ru": "Контекст"},
    "m_feedback": {"en": "Likes / dislikes", "hy": "Հավանել / չհավանել", "ru": "Лайки / дизлайки"},
    "m_scored": {"en": "Outfits evaluated", "hy": "Գնահատված համադրություններ", "ru": "Оценено образов"},
    # today
    "todays_look": {"en": "Today's look", "hy": "Այսօրվա կերպարը", "ru": "Образ дня"},
    "alternatives": {"en": "Alternatives", "hy": "Այլընտրանքներ", "ru": "Альтернативы"},
    "look_n": {"en": "Look {n}", "hy": "Կերպար {n}", "ru": "Образ {n}"},
    "why": {"en": "Why this works", "hy": "Ինչու է սա աշխատում", "ru": "Почему это работает"},
    "score_label": {"en": "Outfit score", "hy": "Գնահատական", "ru": "Оценка образа"},
    "breakdown": {"en": "Score breakdown", "hy": "Գնահատականի բաղադրիչներ", "ru": "Из чего состоит оценка"},
    "like": {"en": "Like", "hy": "Հավանել", "ru": "Нравится"},
    "dislike": {"en": "Dislike", "hy": "Չհավանել", "ru": "Не нравится"},
    "liked_toast": {"en": "Liked · score {a} → {b}", "hy": "Հավանված է · գնահատական {a} → {b}",
                    "ru": "Нравится · оценка {a} → {b}"},
    "disliked_toast": {"en": "Disliked · score {a} → {b}", "hy": "Չհավանված է · գնահատական {a} → {b}",
                       "ru": "Не нравится · оценка {a} → {b}"},
    "no_outfits": {"en": "No outfits yet. Add the missing categories in Wardrobe, or reset the demo.",
                   "hy": "Համադրություններ դեռ չկան։ Ավելացրեք պակասող կատեգորիաները զգեստապահարանում կամ վերագործարկեք դեմոն։",
                   "ru": "Пока нет образов. Добавьте недостающие категории в гардероб или сбросьте демо."},
    # stylist
    "stylist_intro": {"en": "Describe your day in your own words. The stylist picks only from your wardrobe, and every pick is validated and re-scored by the transparent engine.",
                      "hy": "Նկարագրեք ձեր օրը ձեր բառերով։ Ստիլիստն ընտրում է միայն ձեր զգեստապահարանից, իսկ յուրաքանչյուր ընտրություն ստուգվում և վերագնահատվում է թափանցիկ համակարգով։",
                      "ru": "Опишите свой день своими словами. Стилист выбирает только из вашего гардероба, а каждый выбор проверяется и переоценивается прозрачным движком."},
    "chat_placeholder": {"en": "e.g. Dinner date tonight, around 8°C, something classic",
                         "hy": "օր.՝ Այսօր ժամադրություն ունեմ, մոտ 8°C, ուզում եմ դասական ոճ",
                         "ru": "напр.: Свидание вечером, около 8°C, хочу что-то классическое"},
    "q1": {"en": "Dinner date, 8°C, classic", "hy": "Ժամադրություն, 8°C, դասական", "ru": "Свидание, 8°C, классика"},
    "q1_text": {"en": "I have a dinner date tonight, around 8 degrees. Something classic please.",
                "hy": "Այսօր երեկոյան ժամադրություն ունեմ, մոտ 8 աստիճան է։ Խնդրում եմ դասական ինչ-որ բան։",
                "ru": "Сегодня вечером свидание, около 8 градусов. Хочу что-нибудь классическое."},
    "q2": {"en": "Client meeting, 22°C", "hy": "Հանդիպում հաճախորդի հետ, 22°C", "ru": "Встреча с клиентом, 22°C"},
    "q2_text": {"en": "Important client meeting tomorrow, it'll be about 22°C and sunny.",
                "hy": "Վաղը կարևոր հանդիպում ունեմ հաճախորդի հետ, կլինի մոտ 22°C և արևոտ։",
                "ru": "Завтра важная встреча с клиентом, будет около 22°C и солнечно."},
    "q3": {"en": "Lazy Sunday, cold", "hy": "Հանգիստ կիրակի, ցուրտ", "ru": "Ленивое воскресенье, холодно"},
    "q3_text": {"en": "Lazy Sunday coffee with friends, it's cold outside, I want to feel relaxed.",
                "hy": "Հանգիստ կիրակի, սուրճ ընկերների հետ, դրսում ցուրտ է, ուզում եմ ազատ զգալ։",
                "ru": "Ленивое воскресенье, кофе с друзьями, на улице холодно, хочу чувствовать себя расслабленно."},
    "clear_chat": {"en": "Clear chat", "hy": "Մաքրել զրույցը", "ru": "Очистить чат"},
    "styling": {"en": "Styling...", "hy": "Ընտրում եմ...", "ru": "Подбираю..."},
    "src_claude": {"en": "Claude", "hy": "Claude", "ru": "Claude"},
    "src_local": {"en": "free local model", "hy": "անվճար տեղային մոդել", "ru": "бесплатная локальная модель"},
    "src_rules": {"en": "Rule engine", "hy": "Կանոնների համակարգ", "ru": "Движок правил"},
    "engine_score": {"en": "engine score", "hy": "համակարգի գնահատական", "ru": "оценка движка"},
    # wardrobe
    "all": {"en": "All", "hy": "Բոլորը", "ru": "Все"},
    "empty_wardrobe": {"en": "Your wardrobe is empty. Add an item below or reset the demo.",
                       "hy": "Ձեր զգեստապահարանը դատարկ է։ Ավելացրեք իր ներքևում կամ վերագործարկեք դեմոն։",
                       "ru": "Ваш гардероб пуст. Добавьте вещь ниже или сбросьте демо."},
    "remove": {"en": "Remove", "hy": "Հեռացնել", "ru": "Удалить"},
    "removed": {"en": "Item removed.", "hy": "Իրը հեռացված է։", "ru": "Вещь удалена."},
    "photo_credits": {"en": "Photo credits (openly licensed, adapted)", "hy": "Լուսանկարների հեղինակներ (բաց լիցենզիա, փոփոխված)",
                      "ru": "Авторы фото (открытые лицензии, изменены)"},
    "add_item": {"en": "Add an item", "hy": "Ավելացնել իր", "ru": "Добавить вещь"},
    "add_hint": {"en": "Upload a photo: the AI cuts it out, recognises the garment and pre-fills the form. You confirm before saving.",
                 "hy": "Վերբեռնեք լուսանկար. AI-ն կհեռացնի ֆոնը, կճանաչի հագուստը և կլրացնի ձևը։ Դուք հաստատում եք պահպանելուց առաջ։",
                 "ru": "Загрузите фото: AI вырежет фон, распознает вещь и заполнит форму. Вы подтверждаете перед сохранением."},
    "photo": {"en": "Photo (optional) · JPEG, PNG or WebP · max 5 MB", "hy": "Լուսանկար (ըստ ցանկության) · JPEG, PNG կամ WebP · մինչև 5 ՄԲ",
              "ru": "Фото (необязательно) · JPEG, PNG или WebP · до 5 МБ"},
    "analysing": {"en": "Analysing photo...", "hy": "Վերլուծում եմ լուսանկարը...", "ru": "Анализирую фото..."},
    "original": {"en": "Original", "hy": "Բնօրինակ", "ru": "Оригинал"},
    "cutout": {"en": "Background removed", "hy": "Ֆոնը հեռացված է", "ru": "Фон удалён"},
    "as_is": {"en": "Used as-is", "hy": "Օգտագործված է առանց փոփոխության", "ru": "Без изменений"},
    "ai_suggests": {"en": "AI suggests", "hy": "AI-ն առաջարկում է", "ru": "AI предлагает"},
    "confidence": {"en": "confidence", "hy": "վստահություն", "ru": "уверенность"},
    "review_note": {"en": "Suggestions only: review before saving.", "hy": "Միայն առաջարկներ են. ստուգեք պահպանելուց առաջ։",
                    "ru": "Это лишь предложения: проверьте перед сохранением."},
    "heuristic_note": {"en": "Detected dominant color: {c}. Fill in the rest below.",
                       "hy": "Հայտնաբերված հիմնական գույնը՝ {c}։ Լրացրեք մնացածը ներքևում։",
                       "ru": "Определён основной цвет: {c}. Заполните остальное ниже."},
    "name": {"en": "Name", "hy": "Անվանում", "ru": "Название"},
    "category": {"en": "Category", "hy": "Կատեգորիա", "ru": "Категория"},
    "color": {"en": "Color", "hy": "Գույն", "ru": "Цвет"},
    "style": {"en": "Style", "hy": "Ոճ", "ru": "Стиль"},
    "warmth": {"en": "Warmth", "hy": "Ջերմություն", "ru": "Теплота"},
    "formality": {"en": "Formality", "hy": "Պաշտոնականություն", "ru": "Формальность"},
    "save_item": {"en": "Add to wardrobe", "hy": "Ավելացնել զգեստապահարան", "ru": "Добавить в гардероб"},
    "added": {"en": "Added {name} to your wardrobe.", "hy": "{name}-ը ավելացվեց ձեր զգեստապահարանում։",
              "ru": "«{name}» добавлено в гардероб."},
    # insights
    "likes": {"en": "Likes", "hy": "Հավանումներ", "ru": "Лайки"},
    "dislikes": {"en": "Dislikes", "hy": "Չհավանումներ", "ru": "Дизлайки"},
    "learned_items": {"en": "Items with a learned signal", "hy": "Սովորած ազդանշանով իրեր", "ru": "Вещей с выученным сигналом"},
    "how_title": {"en": "Feedback-driven preference adaptation", "hy": "Արձագանքի վրա հիմնված նախասիրությունների հարմարեցում",
                  "ru": "Адаптация предпочтений на основе отзывов"},
    "how_body": {"en": "1. A Like adds +1 to every item in that outfit; a Dislike adds −1. Votes are stored in SQLite.\n"
                       "2. Item signal = tanh(net votes / {scale}), always between −1 and 1.\n"
                       "3. Personalization = 0.5 + 0.5 × mean(item signals), weighted {w}%. Feedback can move a score by at most ±{cap} points: bounded, explainable, reversible. No neural-network training is involved.",
                 "hy": "1. «Հավանել»-ը +1 է ավելացնում համադրության յուրաքանչյուր իրին, «Չհավանել»-ը՝ −1։ Ձայները պահվում են SQLite-ում։\n"
                       "2. Իրի ազդանշանը = tanh(զուտ ձայներ / {scale}), միշտ −1-ի և 1-ի միջև։\n"
                       "3. Անհատականացում = 0.5 + 0.5 × միջին(իրերի ազդանշաններ), կշիռը՝ {w}%։ Արձագանքը կարող է փոխել գնահատականը առավելագույնը ±{cap} միավորով՝ սահմանափակ, բացատրելի, շրջելի։ Նեյրոնային ցանցի ուսուցում չկա։",
                 "ru": "1. «Нравится» добавляет +1 каждой вещи образа, «Не нравится» — −1. Голоса хранятся в SQLite.\n"
                       "2. Сигнал вещи = tanh(чистые голоса / {scale}), всегда от −1 до 1.\n"
                       "3. Персонализация = 0.5 + 0.5 × среднее(сигналы вещей), вес {w}%. Отзывы могут сдвинуть оценку максимум на ±{cap} балла: ограниченно, объяснимо, обратимо. Обучения нейросети нет."},
    "before_after": {"en": "Before / after your last feedback", "hy": "Վերջին արձագանքից առաջ / հետո",
                     "ru": "До / после вашего последнего отзыва"},
    "before_after_empty": {"en": "Like or dislike a look on the Today tab to see the effect here.",
                           "hy": "Հավանեք կամ չհավանեք կերպարը «Այսօր» բաժնում՝ արդյունքը այստեղ տեսնելու համար։",
                           "ru": "Оцените образ на вкладке «Сегодня», чтобы увидеть эффект здесь."},
    "ranking_before": {"en": "Ranking before", "hy": "Դասակարգումը առաջ", "ru": "Рейтинг до"},
    "ranking_now": {"en": "Ranking now (same context)", "hy": "Դասակարգումը հիմա (նույն համատեքստ)",
                    "ru": "Рейтинг сейчас (тот же контекст)"},
    "vs_before": {"en": "pts vs. before", "hy": "միավոր՝ նախորդի համեմատ", "ru": "балла к прежнему"},
    "item_prefs": {"en": "Learned item preferences", "hy": "Սովորած նախասիրություններ", "ru": "Выученные предпочтения"},
    "no_feedback": {"en": "No feedback yet. Every item starts neutral (signal 0).",
                    "hy": "Արձագանքներ դեռ չկան։ Յուրաքանչյուր իր սկսում է չեզոք (ազդանշան 0)։",
                    "ru": "Отзывов пока нет. Каждая вещь стартует нейтрально (сигнал 0)."},
    "style_affinity": {"en": "Style affinity (net votes)", "hy": "Ոճային նախասիրություն (զուտ ձայներ)",
                       "ru": "Склонность к стилям (чистые голоса)"},
    "history": {"en": "Feedback history", "hy": "Արձագանքների պատմություն", "ru": "История отзывов"},
    "col_item": {"en": "Item", "hy": "Իր", "ru": "Вещь"},
    "col_net": {"en": "Net", "hy": "Զուտ", "ru": "Итог"},
    "col_signal": {"en": "Signal", "hy": "Ազդանշան", "ru": "Сигнал"},
    "col_when": {"en": "When (UTC)", "hy": "Երբ (UTC)", "ru": "Когда (UTC)"},
    "col_vote": {"en": "Vote", "hy": "Ձայն", "ru": "Голос"},
    "col_outfit": {"en": "Outfit", "hy": "Համադրություն", "ru": "Образ"},
    "col_context": {"en": "Context", "hy": "Համատեքստ", "ru": "Контекст"},
    "col_score": {"en": "Score at feedback", "hy": "Գնահատականը արձագանքի պահին", "ru": "Оценка на момент отзыва"},
    "section_error": {"en": "This section hit an unexpected error and was skipped. Everything else still works.",
                      "hy": "Այս բաժնում անսպասելի սխալ առաջացավ և այն բաց թողնվեց։ Մնացածը շարունակում է աշխատել։",
                      "ru": "В этом разделе произошла непредвиденная ошибка, и он пропущен. Всё остальное работает."},
    "footer": {"en": "Independent concept demo inspired by LookRoom.ai · not affiliated · Models: Marqo FashionSigLIP / FashionCLIP (ONNX, CPU), BiRefNet-lite, Qwen3-1.7B (local) · Weather: MET Norway (CC BY 4.0)",
               "hy": "LookRoom.ai-ից ոգեշնչված անկախ կոնցեպտ դեմո · կապված չէ · Մոդելներ՝ Marqo FashionSigLIP / FashionCLIP (ONNX, CPU), BiRefNet-lite, Qwen3-1.7B (local) · Եղանակ՝ MET Norway (CC BY 4.0)",
               "ru": "Независимое концепт-демо по мотивам LookRoom.ai · не аффилировано · Модели: Marqo FashionSigLIP / FashionCLIP (ONNX, CPU), BiRefNet-lite, Qwen3-1.7B (local) · Погода: MET Norway (CC BY 4.0)"},
    # week planner
    "tab_week": {"en": "Plan my week", "hy": "Շաբաթվա պլան", "ru": "План на неделю"},
    "week_intro": {"en": "Seven looks for the week ahead: daily forecast, your plans per day, and no piece repeated on consecutive days.",
                   "hy": "Յոթ կերպար առաջիկա շաբաթվա համար՝ օրական եղանակի կանխատեսում, ձեր պլանները և առանց հաջորդական օրերին նույն իրի կրկնության։",
                   "ru": "Семь образов на неделю вперёд: прогноз по дням, ваши планы и без повторения вещей в соседние дни."},
    "week_source_live": {"en": "Forecast: {city} · MET Norway", "hy": "Կանխատեսում՝ {city} · MET Norway",
                         "ru": "Прогноз: {city} · MET Norway"},
    "week_source_manual": {"en": "No live forecast → using {temp}°C for every day. Turn on Live weather above.",
                           "hy": "Կանխատեսում չկա → ամեն օրվա համար {temp}°C։ Միացրեք «Իրական եղանակ»-ը վերևում։",
                           "ru": "Нет прогноза → {temp}°C на каждый день. Включите «Погода онлайн» выше."},
    "plan_for": {"en": "Plans", "hy": "Պլաններ", "ru": "Планы"},
    "days": {"en": "Mon,Tue,Wed,Thu,Fri,Sat,Sun", "hy": "Երկ,Երք,Չրք,Հնգ,Ուրբ,Շբթ,Կիր", "ru": "Пн,Вт,Ср,Чт,Пт,Сб,Вс"},
    "week_variety": {"en": "{n} different pieces used across the week", "hy": "Շաբաթվա ընթացքում օգտագործված է {n} տարբեր իր",
                     "ru": "За неделю использовано {n} разных вещей"},
    # shop the gap
    "tab_shop": {"en": "Shop the gap", "hy": "Ինչ գնել", "ru": "Что докупить"},
    "shop_intro": {"en": "Which single piece would improve your outfits the most right now? Every real product in the catalog is tested against your wardrobe with the same scoring engine.",
                   "hy": "Ո՞ր մեկ իրը հիմա առավելագույնս կբարելավի ձեր համադրությունները։ Կատալոգի յուրաքանչյուր իրական ապրանք փորձարկվում է ձեր զգեստապահարանի հետ նույն գնահատման համակարգով։",
                   "ru": "Какая одна вещь сильнее всего улучшит ваши образы прямо сейчас? Каждый реальный товар каталога проверяется с вашим гардеробом тем же движком оценки."},
    "shop_gain": {"en": "+{g} pts for today", "hy": "+{g} միավոր այսօրվա համար", "ru": "+{g} балла на сегодня"},
    "shop_pairs": {"en": "Pairs with", "hy": "Համադրվում է", "ru": "Сочетается с"},
    "shop_versatile": {"en": "improves {n}/4 occasions", "hy": "բարելավում է {n}/4 առիթ", "ru": "улучшает {n}/4 поводов"},
    "shop_none": {"en": "Your wardrobe already covers this context well: no product adds more than a fraction of a point.",
                  "hy": "Ձեր զգեստապահարանն արդեն լավ է ծածկում այս համատեքստը։",
                  "ru": "Ваш гардероб уже хорошо покрывает этот контекст."},
    "taste": {"en": "taste match {p}%", "hy": "ճաշակի համընկնում {p}%", "ru": "совпадение со вкусом {p}%"},
    "similar": {"en": "Similar", "hy": "Նմանները", "ru": "Похожие"},
    "similar_note": {"en": "Visual similarity · FashionSigLIP image embeddings", "hy": "Տեսողական նմանություն · FashionSigLIP",
                     "ru": "Визуальное сходство · эмбеддинги FashionSigLIP"},
    "shop_add": {"en": "Add to my wardrobe", "hy": "Ավելացնել իմ զգեստապահարանում", "ru": "Добавить в мой гардероб"},
    "shop_catalog_note": {"en": "{n} real products · demo catalog, no purchases", "hy": "{n} իրական ապրանք · դեմո կատալոգ, առանց գնումների",
                          "ru": "{n} реальных товаров · демо-каталог, без покупок"},
    # recommender messages
    "missing_cats": {"en": "Add at least one {cats} item to get outfit recommendations.",
                     "hy": "Ավելացրեք առնվազն մեկ իր հետևյալ կատեգորիաներից՝ {cats}, որպեսզի ստանաք առաջարկներ։",
                     "ru": "Добавьте хотя бы одну вещь из категорий: {cats}, чтобы получить рекомендации."},
    "cold_no_outer": {"en": "It's cold: add outerwear to your wardrobe for warmer suggestions.",
                      "hy": "Ցուրտ է. ավելացրեք վերնազգեստ՝ ավելի տաք առաջարկների համար։",
                      "ru": "Холодно: добавьте верхнюю одежду в гардероб для более тёплых вариантов."},
    "rules_reply": {"en": "For {occ} at {temp}°C in a {style} mood: {names}. {why}",
                    "hy": "{occ} առիթի համար, {temp}°C, {style} ոճով՝ {names}։ {why}",
                    "ru": "Для повода «{occ}», {temp}°C, в стиле «{style}»: {names}. {why}"},
}

EXPLAIN: dict[str, dict[str, str]] = {
    "occ_good": {"en": "Formality suits {occ}.", "hy": "Պաշտոնականության մակարդակը համապատասխանում է՝ {occ}։",
                 "ru": "Уровень формальности подходит для повода «{occ}»."},
    "occ_ok": {"en": "Acceptable formality for {occ}.", "hy": "Ընդունելի պաշտոնականություն՝ {occ}։",
               "ru": "Допустимая формальность для повода «{occ}»."},
    "occ_bad": {"en": "Formality is a stretch for {occ}.", "hy": "Պաշտոնականությունը այնքան էլ չի համապատասխանում՝ {occ}։",
                "ru": "Формальность не очень подходит для повода «{occ}»."},
    "warm_good": {"en": "Warmth fits {temp}.", "hy": "Ջերմությունը համապատասխանում է {temp} եղանակին։",
                  "ru": "По теплу подходит для {temp}."},
    "warm_good_outer": {"en": "Warmth fits {temp}, layered with the {outer}.",
                        "hy": "Ջերմությունը համապատասխանում է {temp} եղանակին, վրայից՝ {outer}։",
                        "ru": "По теплу подходит для {temp}, сверху — {outer}."},
    "warm_bad": {"en": "Warmth is not ideal for {temp}.", "hy": "Ջերմությունն իդեալական չէ {temp} եղանակի համար։",
                 "ru": "По теплу не идеально для {temp}."},
    "warm_bad_outer": {"en": "Warmth is not ideal for {temp} even with the {outer}.",
                       "hy": "Ջերմությունն իդեալական չէ {temp} եղանակի համար, նույնիսկ վերնազգեստով՝ {outer}։",
                       "ru": "По теплу не идеально для {temp}, даже с вещью «{outer}»."},
    "color_neutral": {"en": "All-neutral palette pairs easily.", "hy": "Չեզոք գունապնակը հեշտությամբ է համադրվում։",
                      "ru": "Нейтральная палитра легко сочетается."},
    "color_clash": {"en": "Some colors compete; weaker color harmony.",
                    "hy": "Որոշ գույներ մրցում են իրար հետ, գունային ներդաշնակությունը թույլ է։",
                    "ru": "Некоторые цвета спорят друг с другом — гармония слабее."},
    "color_palette": {"en": "Colors follow a muted palette pairing.", "hy": "Գույները համադրված են զուսպ գունապնակով։",
                      "ru": "Цвета подобраны в сдержанной палитре."},
    "color_anchor": {"en": "Neutral base anchors the accent color.", "hy": "Չեզոք հիմքը հավասարակշռում է ակցենտային գույնը։",
                     "ru": "Нейтральная база уравновешивает акцентный цвет."},
    "style_good": {"en": "Matches your {style} preference.", "hy": "Համապատասխանում է ձեր ոճին՝ {style}։",
                   "ru": "Соответствует вашему стилю: {style}."},
    "style_mix": {"en": "Mixes styles around {style}.", "hy": "Ոճերի համադրություն՝ {style} ոճի շուրջ։",
                  "ru": "Сочетает стили вокруг стиля «{style}»."},
    "pref_up": {"en": "Boosted: includes items you liked.", "hy": "Բարձրացված է. ներառում է ձեր հավանած իրերը։",
                "ru": "Повышено: есть вещи, которые вам понравились."},
    "pref_down": {"en": "Down-weighted: includes items you disliked.", "hy": "Իջեցված է. ներառում է ձեր չհավանած իրերը։",
                  "ru": "Понижено: есть вещи, которые вам не понравились."},
}


def norm(lang: str | None) -> str:
    return lang if lang in LANGS else DEFAULT_LANG


def t(key: str, lang: str, **kw) -> str:
    entry = S.get(key) or EXPLAIN.get(key)
    if entry is None:
        return key
    text = entry.get(norm(lang)) or entry["en"]
    return text.format(**kw) if kw else text


def label(kind: str, value: str, lang: str) -> str:
    entry = LABELS.get(kind, {}).get(value)
    return (entry.get(norm(lang)) or entry["en"]) if entry else value.title()


@lru_cache(maxsize=1)
def _seed_names() -> dict[str, dict[str, str]]:
    path = Path(__file__).resolve().parent.parent / "data" / "demo_items.json"
    try:
        return {r["id"]: r.get("names", {}) for r in json.loads(path.read_text(encoding="utf-8"))}
    except Exception:
        return {}


def item_name(item, lang: str) -> str:
    """Localized name for sample items; user-added items keep the name the user typed."""
    lang = norm(lang)
    if lang == "en":
        return item.name
    return _seed_names().get(item.id, {}).get(lang) or item.name
