import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';

import 'owner_step_images.dart';

// =============================================================================
// COLORS
// =============================================================================

const Color _darkTeal1 = Color(0xFF063F42);
const Color _darkTeal2 = Color(0xFF023336);
const Color _darkTeal3 = Color(0xFF021F22);

const Color _fieldText = Color(0xFF183636);
const Color _fieldHint = Color(0xFF81908F);
const Color _fieldIcon = Color(0xFF07595C);

const Color _danger = Color(0xFFE25A5A);
const Color _warning = Color(0xFFF2A33A);
const Color _success = Color(0xFF54C49A);

// =============================================================================
// SHARED STEP SHELL
// =============================================================================

class _StepShell extends StatelessWidget {
  final int step;
  final String title;
  final String subtitle;

  final AnimationController anim;
  final Animation<double> fade;
  final Animation<double> slide;

  final Widget child;

  const _StepShell({
    required this.step,
    required this.title,
    required this.subtitle,
    required this.anim,
    required this.fade,
    required this.slide,
    required this.child,
  });

  @override
  Widget build(BuildContext context) {
    final keyboardHeight = MediaQuery.viewInsetsOf(context).bottom;

    return ResponsiveShell(
      child: Scaffold(
        resizeToAvoidBottomInset: true,
        backgroundColor: Colors.transparent,
        body: VideoBackground(
          assetPath: 'assets/images/background.mp4',
          overlayOpacity: 0.28,
          cardMargin: const EdgeInsets.all(14),
          cardRadius: 32,
          child: SafeArea(
            child: SingleChildScrollView(
              physics: const BouncingScrollPhysics(),
              keyboardDismissBehavior: ScrollViewKeyboardDismissBehavior.onDrag,
              padding: EdgeInsets.fromLTRB(18, 12, 18, 24 + keyboardHeight),
              child: AnimatedBuilder(
                animation: anim,
                builder: (context, animatedChild) {
                  return Transform.translate(
                    offset: Offset(0, slide.value),
                    child: Opacity(opacity: fade.value, child: animatedChild),
                  );
                },
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _StepHeader(step: step, title: title, subtitle: subtitle),

                    const SizedBox(height: 18),

                    child,
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// =============================================================================
// STEP HEADER
// =============================================================================

class _StepHeader extends StatelessWidget {
  final int step;
  final String title;
  final String subtitle;

  const _StepHeader({
    required this.step,
    required this.title,
    required this.subtitle,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            InkWell(
              onTap: () => Navigator.maybePop(context),
              borderRadius: BorderRadius.circular(50),
              child: Container(
                width: 42,
                height: 42,
                decoration: BoxDecoration(
                  color: const Color(0xFF02383B).withOpacity(0.56),
                  shape: BoxShape.circle,
                  border: Border.all(color: Colors.white.withOpacity(0.22)),
                ),
                child: const Icon(
                  Icons.arrow_back_rounded,
                  color: Colors.white,
                  size: 20,
                ),
              ),
            ),

            const Spacer(),

            Container(
              padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 6),
              decoration: BoxDecoration(
                color: const Color(0xFF02383B).withOpacity(0.60),
                borderRadius: BorderRadius.circular(17),
                border: Border.all(color: AppTheme.tealMint.withOpacity(0.24)),
              ),
              child: Text(
                'Step $step of 9',
                style: GoogleFonts.dmSans(
                  color: Colors.white,
                  fontSize: 9.5,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
          ],
        ),

        const SizedBox(height: 16),

        _ProgressBar(step: step),

        const SizedBox(height: 16),

        Text(
          title,
          style: GoogleFonts.playfairDisplay(
            color: Colors.white,
            fontSize: 35,
            height: 1.03,
            fontWeight: FontWeight.w700,
          ),
        ),

        const SizedBox(height: 10),

        Text(
          subtitle,
          style: GoogleFonts.dmSans(
            color: Colors.white.withOpacity(0.70),
            fontSize: 12.5,
            height: 1.4,
            fontWeight: FontWeight.w500,
          ),
        ),
      ],
    );
  }
}

// =============================================================================
// PROGRESS BAR
// =============================================================================

class _ProgressBar extends StatelessWidget {
  final int step;

  const _ProgressBar({required this.step});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: List.generate(9, (index) {
        final active = index < step;

        return Expanded(
          child: Container(
            height: active ? 4 : 3,
            margin: EdgeInsets.only(right: index < 8 ? 5 : 0),
            decoration: BoxDecoration(
              color: active
                  ? AppTheme.tealMint
                  : Colors.white.withOpacity(0.22),
              borderRadius: BorderRadius.circular(3),
            ),
          ),
        );
      }),
    );
  }
}

// =============================================================================
// MAIN DARK GLASS CARD
// =============================================================================

class _GlassStepCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final List<Widget> children;

  const _GlassStepCard({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.children,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(18, 16, 18, 20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            _darkTeal1.withOpacity(0.90),
            _darkTeal2.withOpacity(0.90),
            _darkTeal3.withOpacity(0.93),
          ],
        ),
        borderRadius: BorderRadius.circular(28),
        border: Border.all(
          color: AppTheme.tealMint.withOpacity(0.22),
          width: 0.9,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.28),
            blurRadius: 26,
            offset: const Offset(0, 11),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Center(
            child: Container(
              width: 38,
              height: 4,
              margin: const EdgeInsets.only(bottom: 16),
              decoration: BoxDecoration(
                color: AppTheme.tealMint.withOpacity(0.30),
                borderRadius: BorderRadius.circular(3),
              ),
            ),
          ),

          Row(
            children: [
              Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  color: AppTheme.teal.withOpacity(0.22),
                  borderRadius: BorderRadius.circular(13),
                  border: Border.all(
                    color: AppTheme.tealMint.withOpacity(0.18),
                  ),
                ),
                child: Icon(icon, color: AppTheme.tealMint, size: 21),
              ),

              const SizedBox(width: 11),

              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.playfairDisplay(
                        color: Colors.white,
                        fontSize: 19,
                        fontWeight: FontWeight.w700,
                      ),
                    ),

                    const SizedBox(height: 3),

                    Text(
                      subtitle,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.dmSans(
                        color: Colors.white.withOpacity(0.50),
                        fontSize: 8.8,
                        height: 1.35,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),

          const SizedBox(height: 18),

          ...children,
        ],
      ),
    );
  }
}

// =============================================================================
// SHARED SECTION LABEL
// =============================================================================

Widget _label(String text, {IconData? icon}) {
  return Padding(
    padding: const EdgeInsets.only(bottom: 9),
    child: Row(
      children: [
        if (icon != null) ...[
          Container(
            width: 27,
            height: 27,
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(0.16),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Icon(icon, size: 14, color: AppTheme.tealMint),
          ),
          const SizedBox(width: 8),
        ],

        Expanded(
          child: Text(
            text,
            style: GoogleFonts.dmSans(
              color: Colors.white,
              fontSize: 11,
              fontWeight: FontWeight.w800,
            ),
          ),
        ),
      ],
    ),
  );
}

// =============================================================================
// YES / NO CONTROL
// =============================================================================

Widget _yesNo(String label, bool value, ValueChanged<bool> onChange) {
  return Padding(
    padding: const EdgeInsets.only(bottom: 11),
    child: LayoutBuilder(
      builder: (context, constraints) {
        if (constraints.maxWidth < 280) {
          return Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: GoogleFonts.dmSans(
                  color: Colors.white.withOpacity(0.82),
                  fontSize: 10.5,
                  fontWeight: FontWeight.w600,
                ),
              ),

              const SizedBox(height: 8),

              Row(
                children: [
                  Expanded(
                    child: _choiceButton(
                      text: 'Yes',
                      selected: value,
                      onTap: () => onChange(true),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: _choiceButton(
                      text: 'No',
                      selected: !value,
                      onTap: () => onChange(false),
                    ),
                  ),
                ],
              ),
            ],
          );
        }

        return Row(
          children: [
            Expanded(
              child: Text(
                label,
                style: GoogleFonts.dmSans(
                  color: Colors.white.withOpacity(0.82),
                  fontSize: 10.5,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),

            const SizedBox(width: 10),

            _choiceButton(
              text: 'Yes',
              selected: value,
              onTap: () => onChange(true),
            ),

            const SizedBox(width: 7),

            _choiceButton(
              text: 'No',
              selected: !value,
              onTap: () => onChange(false),
            ),
          ],
        );
      },
    ),
  );
}

Widget _choiceButton({
  required String text,
  required bool selected,
  required VoidCallback onTap,
}) {
  return InkWell(
    onTap: onTap,
    borderRadius: BorderRadius.circular(10),
    child: AnimatedContainer(
      duration: const Duration(milliseconds: 160),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      decoration: BoxDecoration(
        color: selected
            ? AppTheme.teal
            : const Color(0xFF011F21).withOpacity(0.58),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(
          color: selected
              ? AppTheme.tealMint.withOpacity(0.42)
              : Colors.white.withOpacity(0.12),
        ),
      ),
      child: Text(
        text,
        textAlign: TextAlign.center,
        style: GoogleFonts.dmSans(
          color: selected ? Colors.white : Colors.white.withOpacity(0.58),
          fontSize: 9.5,
          fontWeight: FontWeight.w700,
        ),
      ),
    ),
  );
}

// =============================================================================
// CHECK OPTION
// =============================================================================

Widget _check(
  String label,
  bool value,
  ValueChanged<bool> onChange, {
  IconData? icon,
}) {
  return InkWell(
    onTap: () => onChange(!value),
    borderRadius: BorderRadius.circular(12),
    child: Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 7),
      padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 10),
      decoration: BoxDecoration(
        color: value
            ? AppTheme.teal.withOpacity(0.16)
            : const Color(0xFF011F21).withOpacity(0.48),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: value
              ? AppTheme.tealMint.withOpacity(0.44)
              : Colors.white.withOpacity(0.10),
        ),
      ),
      child: Row(
        children: [
          AnimatedContainer(
            duration: const Duration(milliseconds: 160),
            width: 21,
            height: 21,
            decoration: BoxDecoration(
              color: value ? AppTheme.teal : Colors.transparent,
              borderRadius: BorderRadius.circular(6),
              border: Border.all(
                color: value
                    ? AppTheme.tealMint
                    : Colors.white.withOpacity(0.24),
              ),
            ),
            child: value
                ? const Icon(Icons.check_rounded, color: Colors.white, size: 13)
                : null,
          ),

          const SizedBox(width: 9),

          if (icon != null) ...[
            Icon(icon, size: 15, color: AppTheme.tealMint),
            const SizedBox(width: 7),
          ],

          Expanded(
            child: Text(
              label,
              style: GoogleFonts.dmSans(
                color: Colors.white.withOpacity(value ? 0.90 : 0.68),
                fontSize: 10.3,
                fontWeight: value ? FontWeight.w700 : FontWeight.w500,
              ),
            ),
          ),
        ],
      ),
    ),
  );
}

// =============================================================================
// WHITE INPUT FIELD
// =============================================================================

Widget _field(
  TextEditingController ctrl,
  String hint,
  IconData icon, {
  int maxLines = 1,
  TextInputType? type,
  TextInputAction? action,
  ValueChanged<String>? onSubmitted,
}) {
  return Container(
    width: double.infinity,
    decoration: BoxDecoration(
      color: Colors.white.withOpacity(0.94),
      borderRadius: BorderRadius.circular(14),
      border: Border.all(color: Colors.white.withOpacity(0.30)),
    ),
    child: TextField(
      controller: ctrl,
      maxLines: maxLines,
      keyboardType: type,
      textInputAction: action,
      onSubmitted: onSubmitted,
      style: GoogleFonts.dmSans(
        color: _fieldText,
        fontSize: 12.5,
        fontWeight: FontWeight.w500,
      ),
      decoration: InputDecoration(
        hintText: hint,
        hintStyle: GoogleFonts.dmSans(color: _fieldHint, fontSize: 10.5),
        prefixIcon: Icon(icon, size: 18, color: _fieldIcon),
        border: InputBorder.none,
        contentPadding: const EdgeInsets.symmetric(
          horizontal: 14,
          vertical: 14,
        ),
      ),
    ),
  );
}

// =============================================================================
// DIVIDER
// =============================================================================

Widget _divider() {
  return Padding(
    padding: const EdgeInsets.symmetric(vertical: 7),
    child: Divider(
      height: 1,
      thickness: 0.7,
      color: Colors.white.withOpacity(0.10),
    ),
  );
}

// =============================================================================
// INFO BOX
// =============================================================================

Widget _infoBox(String text) {
  return Container(
    width: double.infinity,
    padding: const EdgeInsets.all(12),
    decoration: BoxDecoration(
      color: AppTheme.teal.withOpacity(0.12),
      borderRadius: BorderRadius.circular(13),
      border: Border.all(color: AppTheme.tealMint.withOpacity(0.20)),
    ),
    child: Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Icon(
          Icons.info_outline_rounded,
          color: AppTheme.tealMint,
          size: 17,
        ),

        const SizedBox(width: 9),

        Expanded(
          child: Text(
            text,
            style: GoogleFonts.dmSans(
              color: Colors.white.withOpacity(0.68),
              fontSize: 9.5,
              height: 1.45,
            ),
          ),
        ),
      ],
    ),
  );
}

// =============================================================================
// NEXT BUTTON
// =============================================================================

Widget _nextBtn(
  BuildContext context,
  String label,
  IconData icon,
  Widget next,
) {
  return SizedBox(
    width: double.infinity,
    height: 50,
    child: ElevatedButton(
      onPressed: () {
        FocusScope.of(context).unfocus();

        Navigator.push(context, MaterialPageRoute(builder: (_) => next));
      },
      style: ElevatedButton.styleFrom(
        backgroundColor: AppTheme.teal,
        foregroundColor: Colors.white,
        elevation: 0,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Flexible(
            child: Text(
              label,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: GoogleFonts.dmSans(
                fontSize: 13,
                fontWeight: FontWeight.w800,
              ),
            ),
          ),

          const SizedBox(width: 9),

          Container(
            width: 25,
            height: 25,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.16),
              shape: BoxShape.circle,
            ),
            child: Icon(icon, size: 14),
          ),
        ],
      ),
    ),
  );
}

// =============================================================================
// ADD BUTTON
// =============================================================================

Widget _addButton(VoidCallback onTap) {
  return InkWell(
    onTap: onTap,
    borderRadius: BorderRadius.circular(14),
    child: Container(
      width: 48,
      height: 48,
      decoration: BoxDecoration(
        color: AppTheme.teal,
        borderRadius: BorderRadius.circular(14),
        boxShadow: [
          BoxShadow(
            color: AppTheme.teal.withOpacity(0.20),
            blurRadius: 12,
            offset: const Offset(0, 5),
          ),
        ],
      ),
      child: const Icon(Icons.add_rounded, color: Colors.white, size: 21),
    ),
  );
}

// =============================================================================
// ADDED ITEM
// =============================================================================

Widget _addedItem({
  required String text,
  required VoidCallback onDelete,
  IconData icon = Icons.check_circle_outline_rounded,
}) {
  return Container(
    width: double.infinity,
    margin: const EdgeInsets.only(bottom: 7),
    padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 9),
    decoration: BoxDecoration(
      color: AppTheme.teal.withOpacity(0.13),
      borderRadius: BorderRadius.circular(11),
      border: Border.all(color: AppTheme.tealMint.withOpacity(0.20)),
    ),
    child: Row(
      children: [
        Icon(icon, color: AppTheme.tealMint, size: 15),

        const SizedBox(width: 8),

        Expanded(
          child: Text(
            text,
            style: GoogleFonts.dmSans(
              color: Colors.white.withOpacity(0.82),
              fontSize: 9.8,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),

        InkWell(
          onTap: onDelete,
          borderRadius: BorderRadius.circular(20),
          child: const Padding(
            padding: EdgeInsets.all(3),
            child: Icon(
              Icons.close_rounded,
              color: AppTheme.tealMint,
              size: 15,
            ),
          ),
        ),
      ],
    ),
  );
}

// =============================================================================
// STEP 6 — MESS & FOOD
// =============================================================================

class OwnerStepMess extends StatefulWidget {
  final String ownerEmail;
  final String ownerName;
  final String hostelName;
  final String hostelType;
  final String city;
  final String address;

  final double latitude;
  final double longitude;
  final int totalFloors;
  final int totalRooms;
  final int totalCapacity;
  final List<Map<String, dynamic>> roomTypes;
  final List<String> facilities;

  const OwnerStepMess({
    super.key,
    required this.ownerEmail,
    required this.ownerName,
    required this.hostelName,
    required this.hostelType,
    required this.city,
    required this.address,
    required this.latitude,
    required this.longitude,
    required this.totalFloors,
    required this.totalRooms,
    required this.totalCapacity,
    required this.roomTypes,
    required this.facilities,
  });

  @override
  State<OwnerStepMess> createState() => _OwnerStepMessState();
}

class _OwnerStepMessState extends State<OwnerStepMess>
    with SingleTickerProviderStateMixin {
  bool _messIncluded = true;
  bool _extraCharges = false;

  int _mealsPerDay = 3;

  String _mealType = 'Both';

  final _chargesCtrl = TextEditingController();

  final _menuCtrl = TextEditingController();

  final _timingCtrl = TextEditingController();

  final List<String> _menu = [];

  final List<String> _timings = [];

  late final AnimationController _anim;
  late final Animation<double> _fade;
  late final Animation<double> _slide;

  @override
  void initState() {
    super.initState();

    _anim = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 750),
    );

    _fade = CurvedAnimation(parent: _anim, curve: Curves.easeOut);

    _slide = Tween<double>(
      begin: 30,
      end: 0,
    ).animate(CurvedAnimation(parent: _anim, curve: Curves.easeOutCubic));

    _anim.forward();
  }

  @override
  void dispose() {
    _anim.dispose();

    _chargesCtrl.dispose();
    _menuCtrl.dispose();
    _timingCtrl.dispose();

    super.dispose();
  }

  void _addTiming() {
    final value = _timingCtrl.text.trim();

    if (value.isEmpty) return;

    if (_timings.any((item) => item.toLowerCase() == value.toLowerCase())) {
      return;
    }

    setState(() {
      _timings.add(value);
      _timingCtrl.clear();
    });

    FocusScope.of(context).unfocus();
  }

  void _addMenuItem() {
    final value = _menuCtrl.text.trim();

    if (value.isEmpty) return;

    if (_menu.any((item) => item.toLowerCase() == value.toLowerCase())) {
      return;
    }

    setState(() {
      _menu.add(value);
      _menuCtrl.clear();
    });

    FocusScope.of(context).unfocus();
  }

  @override
  Widget build(BuildContext context) {
    return _StepShell(
      step: 6,
      title: 'Mess & Food\nDetails',
      subtitle: 'Set up food arrangements for your residents.',
      anim: _anim,
      fade: _fade,
      slide: _slide,
      child: _GlassStepCard(
        icon: Icons.restaurant_rounded,
        title: 'Food Arrangement',
        subtitle: 'Configure mess, meals and timings',
        children: [
          _yesNo('Is mess included in rent?', _messIncluded, (value) {
            setState(() {
              _messIncluded = value;

              if (value) {
                _extraCharges = false;
              }
            });
          }),

          if (!_messIncluded) ...[
            _yesNo('Additional mess charges available?', _extraCharges, (
              value,
            ) {
              setState(() {
                _extraCharges = value;
              });
            }),

            if (_extraCharges) ...[
              _field(
                _chargesCtrl,
                'Monthly mess charges (PKR)',
                Icons.payments_outlined,
                type: TextInputType.number,
              ),

              const SizedBox(height: 13),
            ],
          ],

          _divider(),

          const SizedBox(height: 9),

          _label('Meals per day', icon: Icons.restaurant_rounded),

          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(vertical: 13),
            decoration: BoxDecoration(
              color: const Color(0xFF011F21).withOpacity(0.48),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: Colors.white.withOpacity(0.10)),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _counterButton(Icons.remove_rounded, () {
                  if (_mealsPerDay > 1) {
                    setState(() {
                      _mealsPerDay--;
                    });
                  }
                }),

                const SizedBox(width: 28),

                Column(
                  children: [
                    Text(
                      '$_mealsPerDay',
                      style: GoogleFonts.dmSans(
                        color: AppTheme.tealMint,
                        fontSize: 25,
                        fontWeight: FontWeight.w800,
                      ),
                    ),

                    Text(
                      _mealsPerDay == 1 ? 'meal' : 'meals',
                      style: GoogleFonts.dmSans(
                        color: Colors.white.withOpacity(0.42),
                        fontSize: 8,
                      ),
                    ),
                  ],
                ),

                const SizedBox(width: 28),

                _counterButton(Icons.add_rounded, () {
                  if (_mealsPerDay < 5) {
                    setState(() {
                      _mealsPerDay++;
                    });
                  }
                }),
              ],
            ),
          ),

          const SizedBox(height: 16),

          _label('Meal Type', icon: Icons.set_meal_rounded),

          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: ['Veg Only', 'Non-Veg Only', 'Both'].map((type) {
              final selected = _mealType == type;

              return InkWell(
                onTap: () {
                  setState(() {
                    _mealType = type;
                  });
                },
                borderRadius: BorderRadius.circular(12),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 160),
                  padding: const EdgeInsets.symmetric(
                    horizontal: 13,
                    vertical: 9,
                  ),
                  decoration: BoxDecoration(
                    color: selected
                        ? AppTheme.teal
                        : const Color(0xFF011F21).withOpacity(0.48),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: selected
                          ? AppTheme.tealMint.withOpacity(0.45)
                          : Colors.white.withOpacity(0.10),
                    ),
                  ),
                  child: Text(
                    type,
                    style: GoogleFonts.dmSans(
                      color: selected
                          ? Colors.white
                          : Colors.white.withOpacity(0.62),
                      fontSize: 9.5,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
              );
            }).toList(),
          ),

          const SizedBox(height: 17),

          _label('Meal Timings', icon: Icons.schedule_rounded),

          if (_timings.isNotEmpty) ...[
            ..._timings.map(
              (timing) => _addedItem(
                text: timing,
                icon: Icons.schedule_rounded,
                onDelete: () {
                  setState(() {
                    _timings.remove(timing);
                  });
                },
              ),
            ),

            const SizedBox(height: 3),
          ],

          _responsiveAddRow(
            context: context,
            field: _field(
              _timingCtrl,
              'e.g. Breakfast: 8 AM',
              Icons.access_time_rounded,
              action: TextInputAction.done,
              onSubmitted: (_) => _addTiming(),
            ),
            onAdd: _addTiming,
          ),

          const SizedBox(height: 17),

          _label('Menu Items', icon: Icons.menu_book_rounded),

          Text(
            'Optional — add example meals offered by your mess.',
            style: GoogleFonts.dmSans(
              color: Colors.white.withOpacity(0.42),
              fontSize: 8.5,
            ),
          ),

          const SizedBox(height: 9),

          if (_menu.isNotEmpty) ...[
            Wrap(
              spacing: 7,
              runSpacing: 7,
              children: _menu.map((item) {
                return Container(
                  padding: const EdgeInsets.fromLTRB(10, 7, 5, 7),
                  decoration: BoxDecoration(
                    color: _success.withOpacity(0.14),
                    borderRadius: BorderRadius.circular(11),
                    border: Border.all(color: _success.withOpacity(0.35)),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        Icons.restaurant_menu_rounded,
                        color: _success,
                        size: 13,
                      ),

                      const SizedBox(width: 6),

                      Text(
                        item,
                        style: GoogleFonts.dmSans(
                          color: Colors.white,
                          fontSize: 9.5,
                          fontWeight: FontWeight.w600,
                        ),
                      ),

                      const SizedBox(width: 5),

                      InkWell(
                        onTap: () {
                          setState(() {
                            _menu.remove(item);
                          });
                        },
                        child: const Icon(
                          Icons.close_rounded,
                          color: _success,
                          size: 14,
                        ),
                      ),
                    ],
                  ),
                );
              }).toList(),
            ),

            const SizedBox(height: 9),
          ],

          _responsiveAddRow(
            context: context,
            field: _field(
              _menuCtrl,
              'e.g. Biryani',
              Icons.restaurant_menu_rounded,
              action: TextInputAction.done,
              onSubmitted: (_) => _addMenuItem(),
            ),
            onAdd: _addMenuItem,
          ),

          const SizedBox(height: 20),

          _nextBtn(
            context,
            'Next: Safety & Security',
            Icons.security_rounded,
            OwnerStepSafety(
              ownerEmail: widget.ownerEmail,
              ownerName: widget.ownerName,
              hostelName: widget.hostelName,
              hostelType: widget.hostelType,
              city: widget.city,
              address: widget.address,
              latitude: widget.latitude,
              longitude: widget.longitude,
              totalFloors: widget.totalFloors,
              totalRooms: widget.totalRooms,
              totalCapacity: widget.totalCapacity,
              roomTypes: widget.roomTypes,
              facilities: widget.facilities,
              messIncluded: _messIncluded,
              extraMessCharges: _extraCharges,
              messCharges: _extraCharges ? _chargesCtrl.text.trim() : '',
              mealsPerDay: _mealsPerDay,
              mealType: _mealType,
              mealTimings: List<String>.from(_timings),
              menuItems: List<String>.from(_menu),
            ),
          ),
        ],
      ),
    );
  }

  Widget _counterButton(IconData icon, VoidCallback onTap) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(50),
      child: Container(
        width: 42,
        height: 42,
        decoration: BoxDecoration(
          color: AppTheme.teal.withOpacity(0.20),
          shape: BoxShape.circle,
          border: Border.all(color: AppTheme.tealMint.withOpacity(0.24)),
        ),
        child: Icon(icon, size: 18, color: AppTheme.tealMint),
      ),
    );
  }
}

// =============================================================================
// RESPONSIVE ADD ROW
// =============================================================================

Widget _responsiveAddRow({
  required BuildContext context,
  required Widget field,
  required VoidCallback onAdd,
}) {
  return LayoutBuilder(
    builder: (context, constraints) {
      if (constraints.maxWidth < 250) {
        return Column(
          children: [
            field,

            const SizedBox(height: 8),

            SizedBox(
              width: double.infinity,
              height: 46,
              child: ElevatedButton.icon(
                onPressed: onAdd,
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.teal,
                  foregroundColor: Colors.white,
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(13),
                  ),
                ),
                icon: const Icon(Icons.add_rounded, size: 18),
                label: const Text('Add'),
              ),
            ),
          ],
        );
      }

      return Row(
        children: [
          Expanded(child: field),

          const SizedBox(width: 8),

          _addButton(onAdd),
        ],
      );
    },
  );
}

// =============================================================================
// STEP 7 — SAFETY & SECURITY
// =============================================================================

class OwnerStepSafety extends StatefulWidget {
  final String ownerEmail;
  final String ownerName;
  final String hostelName;
  final String hostelType;
  final String city;
  final String address;

  final double latitude;
  final double longitude;
  final int totalFloors;
  final int totalRooms;
  final int totalCapacity;
  final List<Map<String, dynamic>> roomTypes;
  final List<String> facilities;
  final bool messIncluded;
  final bool extraMessCharges;
  final String messCharges;
  final int mealsPerDay;
  final String mealType;
  final List<String> mealTimings;
  final List<String> menuItems;

  const OwnerStepSafety({
    super.key,
    required this.ownerEmail,
    required this.ownerName,
    required this.hostelName,
    required this.hostelType,
    required this.city,
    required this.address,
    required this.latitude,
    required this.longitude,
    required this.totalFloors,
    required this.totalRooms,
    required this.totalCapacity,
    required this.roomTypes,
    required this.facilities,
    required this.messIncluded,
    required this.extraMessCharges,
    required this.messCharges,
    required this.mealsPerDay,
    required this.mealType,
    required this.mealTimings,
    required this.menuItems,
  });

  @override
  State<OwnerStepSafety> createState() => _OwnerStepSafetyState();
}

class _OwnerStepSafetyState extends State<OwnerStepSafety>
    with SingleTickerProviderStateMixin {
  bool _attendance = true;
  bool _entryCard = false;
  bool _cctv = true;
  bool _warden = true;
  bool _hasCurfew = true;

  TimeOfDay _curfew = const TimeOfDay(hour: 22, minute: 0);

  final Map<String, bool> _cctvAreas = {
    'Entrance': true,
    'Rooms': false,
    'Corridors': true,
    'Common Areas': true,
    'Parking': false,
  };

  final Map<String, bool> _guardShifts = {
    'Day': false,
    'Night': true,
    '24 Hours': false,
  };

  late final AnimationController _anim;
  late final Animation<double> _fade;
  late final Animation<double> _slide;

  @override
  void initState() {
    super.initState();

    _anim = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 750),
    );

    _fade = CurvedAnimation(parent: _anim, curve: Curves.easeOut);

    _slide = Tween<double>(
      begin: 30,
      end: 0,
    ).animate(CurvedAnimation(parent: _anim, curve: Curves.easeOutCubic));

    _anim.forward();
  }

  @override
  void dispose() {
    _anim.dispose();

    super.dispose();
  }

  void _setGuardShift(String key, bool value) {
    setState(() {
      if (key == '24 Hours') {
        _guardShifts['24 Hours'] = value;

        if (value) {
          _guardShifts['Day'] = false;

          _guardShifts['Night'] = false;
        }

        return;
      }

      _guardShifts[key] = value;

      if (value) {
        _guardShifts['24 Hours'] = false;
      }
    });
  }

  Future<void> _selectCurfew() async {
    final time = await showTimePicker(
      context: context,
      initialTime: _curfew,
      builder: (context, child) {
        return MediaQuery(
          data: MediaQuery.of(context).copyWith(size: const Size(390, 844)),
          child: Theme(
            data: Theme.of(context).copyWith(
              colorScheme: const ColorScheme.light(
                primary: AppTheme.teal,
                onPrimary: Colors.white,
                onSurface: _fieldText,
              ),
            ),
            child: child!,
          ),
        );
      },
    );

    if (time == null) return;

    setState(() {
      _curfew = time;
    });
  }

  @override
  Widget build(BuildContext context) {
    return _StepShell(
      step: 7,
      title: 'Safety &\nSecurity',
      subtitle: 'Show residents and parents how your hostel stays secure.',
      anim: _anim,
      fade: _fade,
      slide: _slide,
      child: _GlassStepCard(
        icon: Icons.shield_rounded,
        title: 'Security Setup',
        subtitle: 'Configure monitoring and access controls',
        children: [
          _infoBox(
            'Parents pay close attention to safety. A complete security profile helps build trust.',
          ),

          const SizedBox(height: 15),

          _yesNo('Attendance system available?', _attendance, (value) {
            setState(() {
              _attendance = value;
            });
          }),

          _yesNo('Entry card / access system?', _entryCard, (value) {
            setState(() {
              _entryCard = value;
            });
          }),

          _yesNo('CCTV cameras installed?', _cctv, (value) {
            setState(() {
              _cctv = value;
            });
          }),

          if (_cctv) ...[
            const SizedBox(height: 5),

            _label('CCTV Coverage', icon: Icons.videocam_rounded),

            ..._cctvAreas.entries.map((entry) {
              return _check(entry.key, entry.value, (value) {
                setState(() {
                  _cctvAreas[entry.key] = value;
                });
              }, icon: _cctvAreaIcon(entry.key));
            }),
          ],

          _divider(),

          const SizedBox(height: 9),

          _yesNo('Is warden onsite?', _warden, (value) {
            setState(() {
              _warden = value;
            });
          }),

          const SizedBox(height: 4),

          _label('Security Guard Shifts', icon: Icons.security_rounded),

          ..._guardShifts.entries.map((entry) {
            return _check(
              entry.key,
              entry.value,
              (value) {
                _setGuardShift(entry.key, value);
              },
              icon: entry.key == 'Night'
                  ? Icons.dark_mode_outlined
                  : entry.key == 'Day'
                  ? Icons.light_mode_outlined
                  : Icons.all_inclusive_rounded,
            );
          }),

          _divider(),

          const SizedBox(height: 9),

          _yesNo('Is there a curfew time?', _hasCurfew, (value) {
            setState(() {
              _hasCurfew = value;
            });
          }),

          if (_hasCurfew) ...[
            const SizedBox(height: 5),

            _label('Curfew Time', icon: Icons.lock_clock_rounded),

            InkWell(
              onTap: _selectCurfew,
              borderRadius: BorderRadius.circular(14),
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(
                  horizontal: 14,
                  vertical: 13,
                ),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.94),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(
                    color: AppTheme.tealMint.withOpacity(0.40),
                  ),
                ),
                child: Row(
                  children: [
                    Container(
                      width: 34,
                      height: 34,
                      decoration: BoxDecoration(
                        color: AppTheme.teal.withOpacity(0.10),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: const Icon(
                        Icons.access_time_rounded,
                        color: _fieldIcon,
                        size: 18,
                      ),
                    ),

                    const SizedBox(width: 11),

                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Hostel curfew',
                            style: GoogleFonts.dmSans(
                              color: _fieldHint,
                              fontSize: 8.5,
                            ),
                          ),

                          const SizedBox(height: 2),

                          Text(
                            _curfew.format(context),
                            style: GoogleFonts.dmSans(
                              color: _fieldText,
                              fontSize: 14,
                              fontWeight: FontWeight.w800,
                            ),
                          ),
                        ],
                      ),
                    ),

                    const Icon(Icons.edit_rounded, size: 17, color: _fieldIcon),
                  ],
                ),
              ),
            ),
          ],

          const SizedBox(height: 20),

          _nextBtn(
            context,
            'Next: Rules & Policies',
            Icons.rule_rounded,
            OwnerStepRules(
              ownerEmail: widget.ownerEmail,
              ownerName: widget.ownerName,
              hostelName: widget.hostelName,
              hostelType: widget.hostelType,
              city: widget.city,
              address: widget.address,
              latitude: widget.latitude,
              longitude: widget.longitude,
              totalFloors: widget.totalFloors,
              totalRooms: widget.totalRooms,
              totalCapacity: widget.totalCapacity,
              roomTypes: widget.roomTypes,
              facilities: widget.facilities,
              messIncluded: widget.messIncluded,
              extraMessCharges: widget.extraMessCharges,
              messCharges: widget.messCharges,
              mealsPerDay: widget.mealsPerDay,
              mealType: widget.mealType,
              mealTimings: widget.mealTimings,
              menuItems: widget.menuItems,
              attendanceSystem: _attendance,
              entryCardSystem: _entryCard,
              cctvInstalled: _cctv,
              cctvAreas: _cctvAreas.entries
                  .where((e) => e.value)
                  .map((e) => e.key)
                  .toList(),
              wardenOnsite: _warden,
              guardShifts: _guardShifts.entries
                  .where((e) => e.value)
                  .map((e) => e.key)
                  .toList(),
              hasCurfew: _hasCurfew,
              curfewTime: _hasCurfew
                  ? '${_curfew.hour.toString().padLeft(2, '0')}:${_curfew.minute.toString().padLeft(2, '0')}'
                  : '',
            ),
          ),
        ],
      ),
    );
  }

  IconData _cctvAreaIcon(String area) {
    switch (area) {
      case 'Entrance':
        return Icons.login_rounded;

      case 'Rooms':
        return Icons.bed_outlined;

      case 'Corridors':
        return Icons.route_rounded;

      case 'Common Areas':
        return Icons.groups_outlined;

      case 'Parking':
        return Icons.local_parking_rounded;

      default:
        return Icons.videocam_outlined;
    }
  }
}

// =============================================================================
// STEP 8 — RULES & POLICIES
// =============================================================================

class OwnerStepRules extends StatefulWidget {
  final String ownerEmail;
  final String ownerName;
  final String hostelName;
  final String hostelType;
  final String city;
  final String address;

  final double latitude;
  final double longitude;
  final int totalFloors;
  final int totalRooms;
  final int totalCapacity;
  final List<Map<String, dynamic>> roomTypes;
  final List<String> facilities;
  final bool messIncluded;
  final bool extraMessCharges;
  final String messCharges;
  final int mealsPerDay;
  final String mealType;
  final List<String> mealTimings;
  final List<String> menuItems;
  final bool attendanceSystem;
  final bool entryCardSystem;
  final bool cctvInstalled;
  final List<String> cctvAreas;
  final bool wardenOnsite;
  final List<String> guardShifts;
  final bool hasCurfew;
  final String curfewTime;

  const OwnerStepRules({
    super.key,
    required this.ownerEmail,
    required this.ownerName,
    required this.hostelName,
    required this.hostelType,
    required this.city,
    required this.address,
    required this.latitude,
    required this.longitude,
    required this.totalFloors,
    required this.totalRooms,
    required this.totalCapacity,
    required this.roomTypes,
    required this.facilities,
    required this.messIncluded,
    required this.extraMessCharges,
    required this.messCharges,
    required this.mealsPerDay,
    required this.mealType,
    required this.mealTimings,
    required this.menuItems,
    required this.attendanceSystem,
    required this.entryCardSystem,
    required this.cctvInstalled,
    required this.cctvAreas,
    required this.wardenOnsite,
    required this.guardShifts,
    required this.hasCurfew,
    required this.curfewTime,
  });

  @override
  State<OwnerStepRules> createState() => _OwnerStepRulesState();
}

class _OwnerStepRulesState extends State<OwnerStepRules>
    with SingleTickerProviderStateMixin {
  final Map<String, bool> _attendanceTypes = {
    'Register-Based': true,
    'Digital': false,
    'Biometric': false,
  };

  bool _lateEntry = false;
  bool _parentNotif = true;

  final Map<String, bool> _guestPolicy = {
    'Allowed': false,
    'Not Allowed': true,
    'With Permission': false,
  };

  final Map<String, bool> _paymentMethods = {
    'Cash': true,
    'Bank Transfer': false,
    'JazzCash': false,
    'EasyPaisa': false,
  };

  bool _finePolicy = false;

  final _fineCtrl = TextEditingController();

  final _ruleCtrl = TextEditingController();

  final List<String> _rules = [];

  late final AnimationController _anim;
  late final Animation<double> _fade;
  late final Animation<double> _slide;

  @override
  void initState() {
    super.initState();

    _anim = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 750),
    );

    _fade = CurvedAnimation(parent: _anim, curve: Curves.easeOut);

    _slide = Tween<double>(
      begin: 30,
      end: 0,
    ).animate(CurvedAnimation(parent: _anim, curve: Curves.easeOutCubic));

    _anim.forward();
  }

  @override
  void dispose() {
    _anim.dispose();

    _fineCtrl.dispose();
    _ruleCtrl.dispose();

    super.dispose();
  }

  void _selectAttendanceType(String key) {
    setState(() {
      for (final item in _attendanceTypes.keys) {
        _attendanceTypes[item] = false;
      }

      _attendanceTypes[key] = true;
    });
  }

  void _selectGuestPolicy(String key) {
    setState(() {
      for (final item in _guestPolicy.keys) {
        _guestPolicy[item] = false;
      }

      _guestPolicy[key] = true;
    });
  }

  void _addRule() {
    final value = _ruleCtrl.text.trim();

    if (value.isEmpty) return;

    final exists = _rules.any(
      (rule) => rule.toLowerCase() == value.toLowerCase(),
    );

    if (exists) return;

    setState(() {
      _rules.add(value);

      _ruleCtrl.clear();
    });

    FocusScope.of(context).unfocus();
  }

  @override
  Widget build(BuildContext context) {
    return _StepShell(
      step: 8,
      title: 'Rules &\nPolicies',
      subtitle: 'Define hostel regulations, attendance and payment options.',
      anim: _anim,
      fade: _fade,
      slide: _slide,
      child: _GlassStepCard(
        icon: Icons.rule_rounded,
        title: 'Hostel Policies',
        subtitle: 'Set operational rules for residents',
        children: [
          _label('Attendance Type', icon: Icons.how_to_reg_rounded),

          ..._attendanceTypes.entries.map((entry) {
            return _check(entry.key, entry.value, (_) {
              _selectAttendanceType(entry.key);
            }, icon: _attendanceIcon(entry.key));
          }),

          _divider(),

          const SizedBox(height: 9),

          _yesNo('Late entry allowed?', _lateEntry, (value) {
            setState(() {
              _lateEntry = value;
            });
          }),

          _yesNo('Parent notified on absence?', _parentNotif, (value) {
            setState(() {
              _parentNotif = value;
            });
          }),

          _divider(),

          const SizedBox(height: 9),

          _label('Guest Policy', icon: Icons.people_rounded),

          ..._guestPolicy.entries.map((entry) {
            return _check(entry.key, entry.value, (_) {
              _selectGuestPolicy(entry.key);
            }, icon: _guestIcon(entry.key));
          }),

          _divider(),

          const SizedBox(height: 9),

          _label('Payment Methods', icon: Icons.payments_rounded),

          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: _paymentMethods.entries.map((entry) {
              final selected = entry.value;

              return InkWell(
                onTap: () {
                  setState(() {
                    _paymentMethods[entry.key] = !selected;
                  });
                },
                borderRadius: BorderRadius.circular(12),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 160),
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 9,
                  ),
                  decoration: BoxDecoration(
                    color: selected
                        ? AppTheme.teal
                        : const Color(0xFF011F21).withOpacity(0.50),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: selected
                          ? AppTheme.tealMint.withOpacity(0.42)
                          : Colors.white.withOpacity(0.10),
                    ),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        _paymentIcon(entry.key),
                        size: 14,
                        color: selected
                            ? Colors.white
                            : Colors.white.withOpacity(0.48),
                      ),

                      const SizedBox(width: 6),

                      Text(
                        entry.key,
                        style: GoogleFonts.dmSans(
                          color: selected
                              ? Colors.white
                              : Colors.white.withOpacity(0.64),
                          fontSize: 9.5,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ],
                  ),
                ),
              );
            }).toList(),
          ),

          _divider(),

          const SizedBox(height: 9),

          _yesNo('Fine policy for violations?', _finePolicy, (value) {
            setState(() {
              _finePolicy = value;
            });
          }),

          if (_finePolicy) ...[
            const SizedBox(height: 3),

            _field(
              _fineCtrl,
              'e.g. PKR 500 for late entry',
              Icons.gavel_rounded,
              maxLines: 2,
            ),

            const SizedBox(height: 16),
          ],

          _label('Additional Rules', icon: Icons.rule_rounded),

          if (_rules.isNotEmpty) ...[
            ..._rules.map(
              (rule) => _addedItem(
                text: rule,
                icon: Icons.rule_rounded,
                onDelete: () {
                  setState(() {
                    _rules.remove(rule);
                  });
                },
              ),
            ),

            const SizedBox(height: 3),
          ],

          _responsiveAddRow(
            context: context,
            field: _field(
              _ruleCtrl,
              'e.g. No smoking on premises',
              Icons.add_rounded,
              action: TextInputAction.done,
              onSubmitted: (_) => _addRule(),
            ),
            onAdd: _addRule,
          ),

          const SizedBox(height: 20),

          _nextBtn(
            context,
            'Next: Photos & Videos',
            Icons.photo_library_rounded,
            OwnerStepImages(
              ownerEmail: widget.ownerEmail,
              ownerName: widget.ownerName,
              hostelName: widget.hostelName,
              hostelType: widget.hostelType,
              city: widget.city,
              address: widget.address,
              latitude: widget.latitude,
              longitude: widget.longitude,
              totalFloors: widget.totalFloors,
              totalRooms: widget.totalRooms,
              totalCapacity: widget.totalCapacity,
              roomTypes: widget.roomTypes,
              facilities: widget.facilities,
              messIncluded: widget.messIncluded,
              extraMessCharges: widget.extraMessCharges,
              messCharges: widget.messCharges,
              mealsPerDay: widget.mealsPerDay,
              mealType: widget.mealType,
              mealTimings: widget.mealTimings,
              menuItems: widget.menuItems,
              attendanceSystem: widget.attendanceSystem,
              entryCardSystem: widget.entryCardSystem,
              cctvInstalled: widget.cctvInstalled,
              cctvAreas: widget.cctvAreas,
              wardenOnsite: widget.wardenOnsite,
              guardShifts: widget.guardShifts,
              hasCurfew: widget.hasCurfew,
              curfewTime: widget.curfewTime,
              attendanceType: _attendanceTypes.entries
                  .firstWhere((e) => e.value)
                  .key,
              lateEntryAllowed: _lateEntry,
              parentAbsenceNotification: _parentNotif,
              guestPolicy: _guestPolicy.entries.firstWhere((e) => e.value).key,
              paymentMethods: _paymentMethods.entries
                  .where((e) => e.value)
                  .map((e) => e.key)
                  .toList(),
              finePolicy: _finePolicy,
              fineDetails: _finePolicy ? _fineCtrl.text.trim() : '',
              additionalRules: List<String>.from(_rules),
            ),
          ),
        ],
      ),
    );
  }

  IconData _attendanceIcon(String value) {
    switch (value) {
      case 'Digital':
        return Icons.smartphone_rounded;

      case 'Biometric':
        return Icons.fingerprint_rounded;

      default:
        return Icons.menu_book_rounded;
    }
  }

  IconData _guestIcon(String value) {
    switch (value) {
      case 'Allowed':
        return Icons.check_circle_outline_rounded;

      case 'Not Allowed':
        return Icons.block_rounded;

      default:
        return Icons.verified_user_outlined;
    }
  }

  IconData _paymentIcon(String value) {
    switch (value) {
      case 'Cash':
        return Icons.payments_outlined;

      case 'Bank Transfer':
        return Icons.account_balance_outlined;

      case 'JazzCash':
        return Icons.phone_android_rounded;

      case 'EasyPaisa':
        return Icons.smartphone_rounded;

      default:
        return Icons.payments_outlined;
    }
  }
}
